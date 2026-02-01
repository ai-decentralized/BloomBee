from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import sys
from enum import Enum
from itertools import chain
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

from time import perf_counter
import numpy as np

import torch
from async_timeout import timeout
from hivemind import (
    DHT,
    MSGPackSerializer,
    P2PContext,
    PeerID,
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    nested_flatten,
    nested_pack,
    serialize_torch_tensor,
)
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

import bloombee
from bloombee.data_structures import CHAIN_DELIMITER, UID_DELIMITER, Handle, ModuleUID
from bloombee.server.backend import TransformerBackend
from bloombee.server.memory_cache import AllocationFailed
from bloombee.server.block_functions import iterate_rpc_inference, run_rpc_backward, run_rpc_forward
from bloombee.server.task_prioritizer import DummyTaskPrioritizer, TaskPrioritizerBase
from bloombee.server.speculativeTreePruner import PruningMethod, PruningConfig, BloombeePrunerManager
from bloombee.utils.convert_block import QuantType
from bloombee.utils.microbatch_config import (
    is_microbatch_enabled,
    get_micro_batch_size,
    get_current_path,
    log_path_entry as mbpipe_log_path_entry,
    MBPIPE_LOG_PREFIX,
    AsyncOutputBuffer,
    get_timing_tracker,
)
from bloombee.utils.microbatch_schema import (
    RequestContext,
    create_microbatch_queue_item,
    is_microbatch_queue_item,
    MBPIPE_SCHEMA_PREFIX,
)

logger = get_logger(__name__)


# Create dedicated offloading debug logger
import logging
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)

from datetime import datetime, timezone  
# def print_time_now(s):
#     # Get the current time in UTC  
#     current_utc_datetime = datetime.now(timezone.utc)  
#     # Format the datetime to the desired string format  
#     formatted_utc_time = current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S.%f %Z')  
#     print('\t\t\t'+s+" UTC Time: "+ str(formatted_utc_time) )  
    


# Fix pickling protobufs, see https://stackoverflow.com/a/74873028
sys.modules["runtime_pb2"] = runtime_pb2


CACHE_TOKENS_AVAILABLE = "cache_tokens_available"


class Event(Enum):
    NEW_SESSION = 0
    END_SESSION = 1
    PUSH = 2
    SHUTDOWN = 3


# [MBPIPE] File-based micro-batch accumulator that works across ALL processes
# Uses filesystem for cross-process synchronization since multiprocessing.Manager
# doesn't work across independently spawned processes (Hivemind workers)
import threading
import hashlib
import pickle
import fcntl
import os
import tempfile
from dataclasses import dataclass, field
from typing import Set as TypingSet


# [MBPIPE] Streaming decode state for cross-stage overlap
# Tracks state when processing micro-batches as they arrive from upstream
@dataclass
class StreamingDecodeState:
    """State for a streaming decode session where micro-batches are processed as they arrive."""
    session_id: str
    step_id: str
    total_mbs: int                          # Expected total micro-batches
    full_batch_size: int                    # Full batch size across all MBs
    received_mbs: TypingSet[int] = field(default_factory=set)  # Set of received MB indices
    results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)  # mb_idx -> (hidden, keep_indices)
    cache_allocated: bool = False           # Whether KV cache is allocated
    cache_handles: Optional[List] = None    # KV cache handles
    first_mb_metadata: Optional[Dict] = None  # Metadata from first MB for context
    start_time: float = 0.0                 # Start time for timing

# Directory for storing micro-batch accumulator data
_MB_ACCUMULATOR_DIR = os.path.join(tempfile.gettempdir(), "bloombee_mb_accumulator")

def _get_mb_file_path(acc_key: str, mb_idx: int) -> str:
    """Get file path for a micro-batch."""
    # Use hash to create a safe filename
    key_hash = hashlib.md5(acc_key.encode()).hexdigest()[:16]
    return os.path.join(_MB_ACCUMULATOR_DIR, f"{key_hash}_mb{mb_idx}.pkl")

def _get_mb_lock_path(acc_key: str) -> str:
    """Get lock file path for a step."""
    key_hash = hashlib.md5(acc_key.encode()).hexdigest()[:16]
    return os.path.join(_MB_ACCUMULATOR_DIR, f"{key_hash}.lock")

def _store_microbatch_to_file(acc_key: str, mb_idx: int, tensor_bytes: list, metadata: dict) -> int:
    """
    Store a micro-batch to file and return the count of micro-batches for this step.
    Uses file locking for cross-process safety.
    """
    os.makedirs(_MB_ACCUMULATOR_DIR, exist_ok=True)
    
    lock_path = _get_mb_lock_path(acc_key)
    file_path = _get_mb_file_path(acc_key, mb_idx)
    
    # Use file locking for cross-process synchronization
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Store this micro-batch
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'tensor_bytes': tensor_bytes,
                    'metadata': metadata
                }, f)
            
            # Count how many micro-batches exist for this step
            key_hash = hashlib.md5(acc_key.encode()).hexdigest()[:16]
            count = sum(
                1 for fname in os.listdir(_MB_ACCUMULATOR_DIR)
                if fname.startswith(key_hash) and fname.endswith('.pkl')
            )
            return count
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _load_all_microbatches_from_files(acc_key: str, expected_num: int) -> dict:
    """
    Load all micro-batches for a step from files.
    Returns dict mapping mb_idx to (tensor_bytes, metadata).
    """
    lock_path = _get_mb_lock_path(acc_key)
    
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            result = {}
            for mb_idx in range(expected_num):
                file_path = _get_mb_file_path(acc_key, mb_idx)
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        result[mb_idx] = (data['tensor_bytes'], data['metadata'])
            return result
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _cleanup_microbatch_files(acc_key: str, expected_num: int) -> None:
    """Remove all micro-batch files for a step."""
    lock_path = _get_mb_lock_path(acc_key)
    
    try:
        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                for mb_idx in range(expected_num):
                    file_path = _get_mb_file_path(acc_key, mb_idx)
                    if os.path.exists(file_path):
                        os.remove(file_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        # Also remove lock file
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        logger.debug(f"{MBPIPE_LOG_PREFIX} Failed to cleanup files: {e}")


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    module_backends: Dict[ModuleUID, TransformerBackend]

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, TransformerBackend],
        *,
        adapters: Optional[Sequence[str]],
        dht_prefix: str,
        handler_event_queues: Sequence[mp.Queue],
        handler_index: int,
        inference_max_length: int,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        task_prioritizer: TaskPrioritizerBase = DummyTaskPrioritizer(),
        quant_type: QuantType,
        pruner_manager: BloombeePrunerManager,
    ):
        super().__init__(dht, module_backends)
        for module_backend in self.module_backends.values():
            assert isinstance(module_backend, TransformerBackend)
        self.dht_prefix = dht_prefix
        self.adapters = adapters
        self._handler_event_queues = handler_event_queues
        self._handler_index = handler_index
        self._own_event_queue = handler_event_queues[handler_index]
        self._listener_task: Optional[asyncio.Task] = None
        self._session_queues: Dict[str, asyncio.Queue] = {}
        self._session_handlers: Dict[str, int] = {}
        
        # [MBPIPE] Cross-stage pipeline: micro-batch queues for immediate processing
        # Key: (session_id, step_id) -> Queue holding individual micro-batches
        self._mb_queues: Dict[tuple, asyncio.Queue] = {}
        # Key: (session_id, step_id) -> expected number of micro-batches
        self._mb_expected: Dict[tuple, int] = {}
        # Key: (session_id, step_id) -> count of received micro-batches
        self._mb_received: Dict[tuple, int] = {}
        # [MBPIPE] Cross-stage pipeline: request context cache (Step C/D)
        # Key: (session_id, step_id) -> RequestContext with cached mb0 fields
        self._request_contexts: Dict[tuple, RequestContext] = {}
        # Key: (session_id, step_id) -> set of (mb_idx) already processed (idempotency)
        self._mb_processed: Dict[tuple, set] = {}
        # Feature flag for immediate queuing - ENABLED for cross-stage pipeline overlap
        self._enable_immediate_mb_queue = os.environ.get(
            "BLOOMBEE_ENABLE_IMMEDIATE_MB_QUEUE", "1"
        ) == "1"
        
        logger.info(f"{MBPIPE_LOG_PREFIX} Immediate micro-batch queuing: {'ENABLED' if self._enable_immediate_mb_queue else 'disabled'}")
        
        # [MBPIPE] Streaming decode: process micro-batches as they arrive for cross-stage overlap
        # Key: (session_id, step_id) -> StreamingDecodeState
        self._streaming_decode_sessions: Dict[tuple, StreamingDecodeState] = {}



        self.inference_max_length = inference_max_length
        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout
        self._prioritizer = task_prioritizer
        self.quant_type = quant_type
        self.pruner_manager = pruner_manager


    async def add_p2p_handlers(self, *args, **kwargs) -> None:
        if self._listener_task is None:
            # Start listening to our own event queue before we accept any requests
            self._listener_task = asyncio.create_task(self._listen_to_event_queue())
        await super().add_p2p_handlers(*args, **kwargs)

    def shutdown(self):
        if self.is_alive():
            self._outer_pipe.send("_shutdown")
            self._own_event_queue.put((Event.SHUTDOWN, None, None))
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(f"{self.__class__.__name__} failed to shut down gracefully, sending SIGTERM")
                self.terminate()

    async def _gather_inputs(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> Tuple[str, List[torch.Tensor], Dict]:
        block_uid, metadata = None, None

        def _unpack(req: runtime_pb2.ExpertRequest) -> Iterable[runtime_pb2.Tensor]:
            nonlocal block_uid, metadata

            if block_uid is None:
                block_uid = req.uid
            elif block_uid != req.uid:
                raise ValueError("Block uids differ in one request")

            if metadata is None:
                metadata = MSGPackSerializer.loads(req.metadata) if req.metadata else {}

            return req.tensors

        tensors_stream = amap_in_executor(_unpack, requests)
        inputs = await deserialize_tensor_stream(tensors_stream)
        assert isinstance(block_uid, str) and isinstance(metadata, dict)
        return block_uid, inputs, metadata

    async def rpc_inference(
        self,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        context: P2PContext,
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        """Compute a single step of inference using attention cache; update attention cache accordingly."""
        # offload_logger.info(" Start inference request - rpc_inference")
        # print_time_now('')
        async with timeout(self.session_timeout):
            
            try:
                request = await asyncio.wait_for(anext(requests), self.step_timeout)
            except asyncio.TimeoutError:
                self._log_request("rpc_inference.open", None, context, warning="timed out")
                return

            # RPC Debug: Log received request size
            request_tensor_sizes = [len(tensor.buffer) for tensor in request.tensors]
            request_metadata_size = len(request.metadata) if request.metadata else 0
            total_request_size = sum(request_tensor_sizes) + request_metadata_size
            

            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_inference.open", requested_uids, context)
            try:
                start_time = perf_counter()
                
                metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
                end_msg_serial_time = perf_counter()
                # print_time_now('')
                
                requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
                max_length = metadata.get("max_length")
                points = metadata.get("points", 0)
                session_id = metadata.get("session_id")
                alloc_timeout = float(metadata.get("alloc_timeout", 0.0))
                args_structure = metadata.get("args_structure")
                if not requested_uids:
                    raise ValueError("User must specify at least one block for inference, but got none")
                assert isinstance(
                    max_length, int
                ), f"rpc_inference metadata must contain int max_length, got {max_length}"
                assert isinstance(
                    points, (float, int)
                ), f"rpc_inference should have number of points as a number or None, got {points}"
                if not 0 <= max_length <= self.inference_max_length:
                    raise ValueError(
                        f"Cannot allocate KV cache for {max_length} tokens, max = {self.inference_max_length}"
                    )

                original_batch_size = request.tensors[0].size[0] if request.tensors else 1
                batch_size = original_batch_size
                
                # [MB_DEBUG] Log initial batch size detection
                logger.info(f"[MB_DEBUG] === BATCH SIZE DETECTION ===")
                logger.info(f"[MB_DEBUG] Original batch_size from tensor[0]: {original_batch_size}")
                logger.info(f"[MB_DEBUG] Metadata keys: {list(metadata.keys())}")
                logger.info(f"[MB_DEBUG] metadata.type={metadata.get('type')}, metadata.full_batch_size={metadata.get('full_batch_size')}")
                
                # [MBPIPE_STREAMING] For cross-stage micro-batch streaming, use full_batch_size for KV cache allocation
                # This is critical for decode overlap: Stage 2 must allocate cache for full batch on first MB arrival
                is_streaming_decode = is_microbatch_queue_item(request) or metadata.get("type") == "micro_batch"
                logger.info(f"[MB_DEBUG] is_streaming_decode={is_streaming_decode}, is_microbatch_queue_item={is_microbatch_queue_item(request)}")
                
                if is_streaming_decode:
                    streaming_full_batch_size = metadata.get("full_batch_size", batch_size)
                    logger.info(f"[MB_DEBUG] Streaming decode detected! streaming_full_batch_size={streaming_full_batch_size}")
                    if streaming_full_batch_size > batch_size:
                        logger.info(f"{MBPIPE_LOG_PREFIX} [STREAMING_DECODE] Detected streaming micro-batch, "
                                   f"using full_batch_size={streaming_full_batch_size} for KV cache (actual MB size={batch_size})")
                        batch_size = streaming_full_batch_size
                        logger.info(f"[MB_DEBUG] KV cache will use batch_size={batch_size} (overridden from {original_batch_size})")
                else:
                    logger.info(f"[MB_DEBUG] NOT streaming decode, using original batch_size={batch_size}")
                
                end_batch_size_time = perf_counter()
                logger.info(f"[MB_DEBUG] Batch size detection completed in {(end_batch_size_time - perf_counter())*1000:.2f}ms, final batch_size={batch_size}")
                # print_time_now('')
                
                # [MBPIPE] Log current path at rpc_inference entry
                mbpipe_log_path_entry(logger, "handler.rpc_inference", batch_size=batch_size)

                
                push_time = []
                
                # [KVCACHE_DEBUG] Log before cache allocation
                cache_alloc_start = perf_counter()
                logger.info(f"[KVCACHE_DEBUG] === KV CACHE ALLOCATION ===")
                logger.info(f"[KVCACHE_DEBUG] Allocating cache: batch_size={batch_size}, max_length={max_length}, timeout={alloc_timeout}")
                logger.info(f"[KVCACHE_DEBUG] Requested backends: {len(requested_backends)}, UIDs: {requested_uids}")
                
                async with self._allocate_cache(
                    requested_backends, batch_size=batch_size, max_length=max_length, timeout=alloc_timeout
                ) as cache_handles:
                    end_cache_time = perf_counter()
                    cache_alloc_ms = (end_cache_time - cache_alloc_start) * 1000
                    
                    # [KVCACHE_DEBUG] Log cache allocation result
                    logger.info(f"[KVCACHE_DEBUG] Cache allocated in {cache_alloc_ms:.2f}ms")
                    logger.info(f"[KVCACHE_DEBUG] cache_handles count: {len(cache_handles) if cache_handles else 0}")
                    if cache_handles:
                        for i, handles in enumerate(cache_handles):
                            logger.info(f"[KVCACHE_DEBUG] cache_handles[{i}]: {len(handles) if handles else 0} handles")
                    
                    background_tasks = set()
                    step_=0
                    warmup_completed = False  # Track if warmup/prefill phase is completed
                    
                    # [MBPIPE] Async Output Buffer for compute/communication overlap
                    output_buffer: Optional[AsyncOutputBuffer] = None
                    use_buffer = False
                    
                    # Check if we should use async buffer (based on timing data)
                    # Server-to-server communication is determined by next_servers in metadata
                    if is_microbatch_enabled():
                        # Check timing data for buffer decision
                        tracker = get_timing_tracker()
                        use_buffer, buffer_pos = tracker.should_use_buffer()
                        
                        if use_buffer and buffer_pos == "producer":
                            output_buffer = AsyncOutputBuffer(
                                max_pending=2,  # Allow up to 2 pending sends
                                logger=logger,
                                name=f"server_{requested_uids[0]}"
                            )
                            
                            # Define the async push function for the buffer
                            # Must be async because _push_outputs is async
                            async def buffered_push_fn(item):
                                req, tensors, meta = item
                                await self._push_outputs(req, tensors, meta)
                            
                            await output_buffer.start_sender(buffered_push_fn)
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer started for cross-stage overlap"
                            )
                    
                    # [MBPIPE] Cross-stage streaming push callback (for micro-batch level streaming)
                    # This enables Server2 to start processing micro-batch N while Server1 computes N+1
                    cross_stage_push_microbatch = None
                    
                    if is_microbatch_enabled():
                        # Create the cross-stage push function that captures required context
                        async def _cross_stage_push_wrapper(mb_hidden, mb_keep, push_metadata):
                            """Wrapper that calls _push_microbatch with required backends."""
                            await self._push_microbatch(
                                mb_hidden, mb_keep, push_metadata, requested_backends
                            )
                        
                        cross_stage_push_microbatch = _cross_stage_push_wrapper
                        logger.info(f"{MBPIPE_LOG_PREFIX} Cross-stage micro-batch push enabled")
                    
                    # print('before async for output_tensors, can_push, step_metadata in iterate_rpc_inference() ') ###
                    # print_time_now('')
                    # offload_logger.info(" Start inference iteration")
                    async for output_tensors, can_push, step_metadata in iterate_rpc_inference(
                        requested_uids=requested_uids,
                        requested_backends=requested_backends,
                        active_adapter=self._get_active_adapter(metadata),
                        input_iterator=self._iterate_inference_steps(
                            request, requests, session_id, requested_uids, context
                        ),
                        cache_handles=cache_handles,
                        pruner_manager=self.pruner_manager,
                        max_length=max_length,
                        prioritizer=self._prioritizer,
                        points=points,
                        quant_type=self.quant_type,
                        args_structure=args_structure,
                        cross_stage_push_fn=cross_stage_push_microbatch,  # [MBPIPE] Cross-stage streaming (currently disabled)
                    ):
                        # offload_logger.info(f" Inference step {step_}: can_push={can_push}")
                        # print('=================================================   server rpc_inference step ',step_) ###
                        # print_time_now('')
                        step_+=1 ###
                        
                        # After first step (warmup/prefill), clean up temporary shared memory
                        # This helps reduce /dev/shm peak usage on systems with limited shared memory
                        # For larger batch sizes, perform cleanup more aggressively
                        if not warmup_completed and step_ > 0:
                            warmup_completed = True
                            self._cleanup_warmup_shared_memory()
                        # For large batch sizes, also cleanup periodically to prevent accumulation
                        elif step_ > 0 and step_ % 5 == 0 and batch_size >= 20:
                            self._cleanup_warmup_shared_memory()
                        
                        can_push_case_time=perf_counter() ###

                        if can_push:
                            # [MBPIPE] Skip _push_outputs if data was already sent via cross-stage micro-batch push
                            cross_stage_pushed = step_metadata.get("cross_stage_pushed", False) if step_metadata else False
                            if cross_stage_pushed:
                                logger.info(f"{MBPIPE_LOG_PREFIX} Skipping _push_outputs: data sent via cross-stage micro-batch push")
                            elif output_buffer is not None and output_buffer.is_running:
                                # Non-blocking put into buffer - actual send happens in background
                                try:
                                    await output_buffer.put(
                                        (request, output_tensors, step_metadata),
                                        clone=False  # Already serialized, no need to clone
                                    )
                                except Exception as e:
                                    logger.warning(f"{MBPIPE_LOG_PREFIX} Buffer put failed, falling back to direct send: {e}")
                                    task = asyncio.create_task(self._push_outputs(request, output_tensors, step_metadata))
                                    background_tasks.add(task)
                                    task.add_done_callback(background_tasks.discard)
                            else:
                                # Original direct task creation
                                task = asyncio.create_task(self._push_outputs(request, output_tensors, step_metadata))
                                background_tasks.add(task)  # Keep reference until it is done to save it from GC
                                task.add_done_callback(background_tasks.discard)
                        start_ExpertResponse_time=perf_counter() ###
                        push_time.append(start_ExpertResponse_time-can_push_case_time) ###
                        # print('current step push outputs task prepare time ', start_ExpertResponse_time-can_push_case_time) ###
                        # print_time_now('')
                        yield runtime_pb2.ExpertResponse(tensors=output_tensors)
                        end_ExpertResponse_time=perf_counter() ###
                        # print('runtime_pb2.ExpertResponse push outputs respond time', end_ExpertResponse_time-start_ExpertResponse_time) ###
                        # print_time_now('')
                        
                    end_iterate_rpc_inference_time=perf_counter() ###
                    # print('mean push time ', np.mean(push_time[4:])) ###
                    # print('finish iterate_rpc_inference time(sec) ', end_iterate_rpc_inference_time - end_cache_time) ###
                    # print_time_now('')
                    
                    # [MBPIPE] Cleanup async buffer if used
                    if output_buffer is not None:
                        try:
                            await output_buffer.flush()  # Wait for pending sends
                            await output_buffer.stop()   # Stop background sender
                        except Exception as e:
                            logger.warning(f"{MBPIPE_LOG_PREFIX} Buffer cleanup failed: {e}")
            
            finally:
                self._log_request("rpc_inference.close", requested_uids, context)
                # print_time_now('')
                # print('end of  rpc_inference ..........')  ###
                end_time_rpc_infer = perf_counter() ###
                # print('rpc_inference total time(sec) ', end_time_rpc_infer - start_time) ###
            

    @contextlib.contextmanager
    def _managed_session(self, session_id: str):
        assert session_id not in self._session_queues, f"session id {session_id} is not unique"
        try:
            self._session_queues[session_id] = asyncio.Queue()
            self._session_handlers[session_id] = self._handler_index
            for other_index, other_queue in enumerate(self._handler_event_queues):
                if other_index != self._handler_index:
                    other_queue.put_nowait((Event.NEW_SESSION, session_id, self._handler_index))
            yield
        finally:
            self._session_queues.pop(session_id).put_nowait(None)  # put None so that the get task will not hang
            del self._session_handlers[session_id]
            for other_index, other_queue in enumerate(self._handler_event_queues):
                if other_index != self._handler_index:
                    other_queue.put_nowait((Event.END_SESSION, session_id, self._handler_index))

    def _put_into_session_queue(self, session_id: str, request: runtime_pb2.ExpertRequest):
        handler_index = self._session_handlers.get(session_id)
        if handler_index is None:
            logger.debug(f"Ignored rpc_push to unknown session ID: {session_id}")
        elif handler_index == self._handler_index:
            self._session_queues[session_id].put_nowait(request)
        else:
            self._handler_event_queues[handler_index].put_nowait((Event.PUSH, session_id, request))

    async def _get_from_session_queue(self, session_id: str) -> Optional[runtime_pb2.ExpertRequest]:
        assert self._session_handlers[session_id] == self._handler_index, "session belongs to another handler"
        return await self._session_queues[session_id].get()

    async def _listen_to_event_queue(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                event, session_id, payload = await loop.run_in_executor(None, self._own_event_queue.get)
                if event == Event.SHUTDOWN:
                    break
                elif event == Event.NEW_SESSION:
                    self._session_handlers[session_id] = payload  # index of the handler that owns that session
                elif event == Event.END_SESSION:
                    self._session_handlers.pop(session_id, None)
                elif event == Event.PUSH:
                    maybe_session_queue = self._session_queues.get(session_id)
                    if maybe_session_queue is not None:
                        maybe_session_queue.put_nowait(payload)
                else:
                    raise RuntimeError(f"Unexpected event: {event}")
            except Exception as e:
                logger.exception(e)

    async def _poll_microbatch_file(
        self,
        acc_key: str,
        mb_idx: int,
        timeout: float = 5.0,
        poll_interval: float = 0.01,
    ) -> Optional[Tuple[runtime_pb2.ExpertRequest, dict]]:
        """
        [MBPIPE] Poll for a specific micro-batch from file storage.
        
        This is used for cross-stage pipeline where micro-batches are stored
        by one handler and read by another (file-based IPC).
        
        Returns (request, metadata) when available, or None on timeout.
        """
        file_path = _get_mb_file_path(acc_key, mb_idx)
        lock_path = _get_mb_lock_path(acc_key)
        start_time = perf_counter()
        
        while (perf_counter() - start_time) < timeout:
            try:
                if os.path.exists(file_path):
                    with open(lock_path, 'w') as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                        try:
                            if os.path.exists(file_path):
                                with open(file_path, 'rb') as f:
                                    data = pickle.load(f)
                                    tensor_bytes = data['tensor_bytes']
                                    metadata = data['metadata']
                                    
                                # Reconstruct the request
                                tensors = [runtime_pb2.Tensor.FromString(t) for t in tensor_bytes]
                                request = runtime_pb2.ExpertRequest(
                                    uid=acc_key.split('|')[0],  # session_id
                                    tensors=tensors,
                                    metadata=MSGPackSerializer.dumps(metadata),
                                )
                                
                                logger.info(
                                    f"{MBPIPE_LOG_PREFIX} poll_file: mb_idx={mb_idx} found "
                                    f"after {(perf_counter() - start_time)*1000:.1f}ms"
                                )
                                return request, metadata
                        finally:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logger.debug(f"{MBPIPE_LOG_PREFIX} poll_file error: {e}")
            
            await asyncio.sleep(poll_interval)
        
        logger.warning(
            f"{MBPIPE_LOG_PREFIX} poll_file: timeout waiting for mb_idx={mb_idx} "
            f"after {timeout}s"
        )
        return None

    async def _iterate_inference_steps(
        self,
        first_request: runtime_pb2.ExpertRequest,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        session_id: Optional[str],
        requested_uids: Sequence[str],
        context: P2PContext,
    ) -> AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]]:
        processed_step_ids = set()
        n_pushes = n_late_pushes = 0
        request = first_request
        anext_task = get_push_task = None
        try:
            start_iterate_inference_steps_time = perf_counter()
            
            with self._managed_session(session_id) if session_id is not None else contextlib.nullcontext():
                while request is not None:
                    # [MBPIPE] Check if this is a micro-batch queue item (dict with type="micro_batch")
                    if is_microbatch_queue_item(request):
                        # Yield micro-batch directly with type marker
                        mb_item = request
                        mb_metadata = mb_item.get("metadata", {}).copy()
                        mb_metadata["type"] = "micro_batch"
                        mb_metadata["mb_idx"] = mb_item.get("mb_idx", 0)
                        mb_metadata["expected_num_mb"] = mb_item.get("expected_num_mb", 1)
                        mb_metadata["offset"] = mb_item.get("offset", 0)
                        mb_metadata["size"] = mb_item.get("size", 1)
                        mb_metadata["full_batch_size"] = mb_item.get("full_batch_size", 1)
                        mb_metadata["pushed"] = True
                        
                        logger.info(
                            f"{MBPIPE_LOG_PREFIX} iterate_steps: yielding micro-batch "
                            f"mb_idx={mb_item.get('mb_idx')} for immediate processing"
                        )
                        
                        yield mb_item.get("payload"), mb_metadata
                        
                        # Continue to next item from queue
                        request = None
                    elif hasattr(request, 'tensors') and (request.tensors or (request.metadata and not request.tensors)):
                        # Original full-batch request path
                        start_meta_time = perf_counter()
                        metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
                        step_id = metadata.get("step_id")
                        pushed = metadata.get("pushed")
                        
                        # [MBPIPE] Note: Micro-batch signal handling removed.
                        if metadata.get("is_mb_start_signal"):
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} iterate_steps: ignoring mb_start_signal (incompatible format)"
                            )
                            request = runtime_pb2.ExpertRequest()
                            continue
                        
                        if pushed:
                            n_pushes += 1
                            self._log_request("rpc_inference.push", requested_uids, context, debug=f"session received push")

                        if step_id is None or step_id not in processed_step_ids:
                            yield request, metadata
                            if step_id is not None:
                                processed_step_ids.add(step_id)
                        elif pushed:
                            n_late_pushes += 1
                            self._log_request(
                                "rpc_inference.push",
                                requested_uids,
                                context,
                                debug=f"arrived late {n_late_pushes / n_pushes * 100:.1f}% of the time",
                            )
                        
                        request = None  # Mark as processed, will fetch next
                    else:
                        # Empty or None request - break out
                        break
                    
                    # Wait for the next request, coming either from the `requests` iterator or `push_queue`
                    if anext_task is None:
                        anext_task = asyncio.create_task(anext(requests))
                    if get_push_task is None:
                        if session_id is not None:
                            get_push_task = asyncio.create_task(self._get_from_session_queue(session_id))
                        else:
                            get_push_task = asyncio.create_task(asyncio.Event().wait())  # Dummy never-ending task
                    done, _ = await asyncio.wait(
                        [anext_task, get_push_task], timeout=self.step_timeout, return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    if anext_task in done:
                        request = await anext_task
                        anext_task = None
                    elif get_push_task in done:
                        request = await get_push_task
                        get_push_task = None
                    else:
                        self._log_request("rpc_inference.step", requested_uids, context, warning="timed out")
                        anext_task.cancel()
                        get_push_task.cancel()
                        return
        except Exception:
            raise


    async def rpc_push(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        """Directly push activation tensors from one server to another"""

        requested_uids = self._check_uids(request.uid)
        metadata = MSGPackSerializer.loads(request.metadata)
        session_id = metadata["session_id"]
        
        # [MBPIPE] Check if this is a micro-batch push from cross-stage streaming
        is_microbatch_push = metadata.get("is_microbatch_push", False)
        
        if is_microbatch_push:
            # Handle micro-batch push: accumulate until we have all micro-batches
            return await self._handle_microbatch_push(request, metadata, requested_uids, context)
        
        # Original flow: put into session queue for normal processing
        self._log_request("rpc_push", requested_uids, context, debug=f"session_id={session_id}")
        self._put_into_session_queue(session_id, request)
        return runtime_pb2.ExpertResponse()
    
    async def _handle_microbatch_push(
        self,
        request: runtime_pb2.ExpertRequest,
        metadata: dict,
        requested_uids: Sequence[str],
        context: P2PContext,
    ) -> runtime_pb2.ExpertResponse:
        """
        [MBPIPE] Handle a micro-batch push from upstream server.
        
        With immediate queuing enabled (default):
        - Each micro-batch is put directly into session queue
        - consume side detects micro-batch items and processes them individually
        
        With immediate queuing disabled (fallback):
        - Wait for all micro-batches, assemble, then put into session queue
        """
        session_id = metadata["session_id"]
        step_id = metadata.get("step_id")
        mb_idx = metadata.get("micro_batch_idx", 0)
        mb_offset = metadata.get("micro_batch_offset", 0)
        mb_size = metadata.get("micro_batch_size", 1)
        full_batch_size = metadata.get("full_batch_size", mb_size)
        
        # Use total_micro_batches from metadata if available, otherwise calculate
        expected_num_mb = metadata.get("total_micro_batches")
        if expected_num_mb is None:
            micro_batch_size_config = get_micro_batch_size()
            expected_num_mb = (full_batch_size + micro_batch_size_config - 1) // micro_batch_size_config
        
        mb_key = (session_id, step_id)
        
        # [MBPIPE] Idempotency check - skip if already processed
        if mb_key not in self._mb_processed:
            self._mb_processed[mb_key] = set()
        
        if mb_idx in self._mb_processed[mb_key]:
            logger.info(
                f"{MBPIPE_LOG_PREFIX} rpc_push: mb_idx={mb_idx} already processed (idempotency), skipping"
            )
            return runtime_pb2.ExpertResponse()
        
        self._mb_processed[mb_key].add(mb_idx)
        
        # Initialize tracking for this (session, step) if not exists
        if mb_key not in self._mb_queues:
            self._mb_queues[mb_key] = asyncio.Queue()
            self._mb_expected[mb_key] = expected_num_mb
            self._mb_received[mb_key] = 0
            logger.info(
                f"{MBPIPE_LOG_PREFIX} rpc_push: created tracking for step={step_id}, "
                f"expecting {expected_num_mb} micro-batches, "
                f"immediate_queue={'enabled' if self._enable_immediate_mb_queue else 'disabled'}"
            )
        
        self._mb_received[mb_key] = self._mb_received.get(mb_key, 0) + 1
        received_count = self._mb_received[mb_key]
        
        # Store to file for cross-handler communication (needed even with immediate queue)
        acc_key = f"{session_id}|{step_id}"
        tensor_bytes = [t.SerializeToString() for t in request.tensors]
        file_received_count = _store_microbatch_to_file(acc_key, mb_idx, tensor_bytes, metadata.copy())
        
        if self._enable_immediate_mb_queue:
            # ========== IMMEDIATE QUEUING PATH (Step A) ==========
            # Put each micro-batch directly into session queue as a queue item
            
            mb_queue_item = create_microbatch_queue_item(
                request_id=session_id,
                step_id=step_id,
                mb_idx=mb_idx,
                expected_num_mb=expected_num_mb,
                payload=request,
                metadata=metadata.copy(),
                offset=mb_offset,
                size=mb_size,
                full_batch_size=full_batch_size,
            )
            
            # Put into session queue immediately (wrapped in dict to distinguish from regular requests)
            self._put_into_session_queue(session_id, mb_queue_item)
            
            logger.info(
                f"{MBPIPE_LOG_PREFIX} rpc_push: mb_idx={mb_idx} IMMEDIATELY queued to session "
                f"(received={received_count}/{expected_num_mb})"
            )
            
            # Cleanup file storage when all micro-batches received
            if file_received_count >= expected_num_mb:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} rpc_push: all {expected_num_mb} micro-batches queued, "
                    f"cleaning up file storage"
                )
                _cleanup_microbatch_files(acc_key, expected_num_mb)
                # Cleanup tracking data
                self._mb_queues.pop(mb_key, None)
                self._mb_expected.pop(mb_key, None)
                self._mb_received.pop(mb_key, None)
                self._mb_processed.pop(mb_key, None)
                self._request_contexts.pop(mb_key, None)
        else:
            # ========== LEGACY PATH (wait-all-then-assemble) ==========
            logger.info(
                f"{MBPIPE_LOG_PREFIX} rpc_push: mb_idx={mb_idx} stored to file "
                f"(file_count={file_received_count}/{expected_num_mb})"
            )
            
            if file_received_count >= expected_num_mb:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} rpc_push: all {expected_num_mb} micro-batches received, "
                    f"assembling and forwarding to session"
                )
                try:
                    assembled_request = await self._assemble_microbatches(
                        acc_key, expected_num_mb, requested_uids
                    )
                    self._put_into_session_queue(session_id, assembled_request)
                finally:
                    _cleanup_microbatch_files(acc_key, expected_num_mb)
                    # Cleanup tracking data
                    self._mb_queues.pop(mb_key, None)
                    self._mb_expected.pop(mb_key, None)
                    self._mb_received.pop(mb_key, None)
                    self._mb_processed.pop(mb_key, None)
        
        return runtime_pb2.ExpertResponse()

    
    async def _assemble_microbatches(
        self,
        acc_key: str,
        expected_num_mb: int,
        requested_uids: Sequence[str],
    ) -> runtime_pb2.ExpertRequest:
        """
        [MBPIPE] Assemble accumulated micro-batches into a single request.
        
        This reconstructs the original full batch from multiple micro-batch pushes
        stored in files.
        """
        # Load all micro-batches from files
        accumulated_raw = _load_all_microbatches_from_files(acc_key, expected_num_mb)
        
        # Deserialize protobuf tensors
        accumulated = {
            k: (
                [runtime_pb2.Tensor.FromString(t) for t in v[0]],  # Deserialize from bytes
                v[1].copy()
            )
            for k, v in accumulated_raw.items()
        }
        
        # Sort by micro-batch index to maintain order
        sorted_mb_indices = sorted(accumulated.keys())
        
        if len(sorted_mb_indices) != expected_num_mb:
            raise ValueError(
                f"{MBPIPE_LOG_PREFIX} Micro-batch count mismatch: "
                f"expected {expected_num_mb}, got {len(sorted_mb_indices)}"
            )
        
        # Collect tensors from each micro-batch
        # Assume each micro-batch has same number of tensors (hidden_states, keep_indices)
        first_tensors, first_metadata = accumulated[sorted_mb_indices[0]]
        num_tensor_types = len(first_tensors)
        
        # Deserialize and concatenate tensors
        assembled_tensors = []
        for tensor_idx in range(num_tensor_types):
            tensor_parts = []
            for mb_idx in sorted_mb_indices:
                tensors, _ = accumulated[mb_idx]
                if tensor_idx < len(tensors):
                    # Deserialize tensor
                    tensor = deserialize_torch_tensor(tensors[tensor_idx])
                    tensor_parts.append(tensor)
            
            if tensor_parts:
                # Concatenate along batch dimension
                if len(tensor_parts) > 1:
                    assembled = torch.cat(tensor_parts, dim=0)
                else:
                    assembled = tensor_parts[0]
                
                # Re-serialize for the ExpertRequest
                # Get compression settings from original tensor proto
                original_proto = first_tensors[tensor_idx]
                assembled_proto = serialize_torch_tensor(
                    assembled, original_proto.compression, allow_inplace=True
                )
                assembled_tensors.append(assembled_proto)
        
        # Build assembled metadata (use first micro-batch's metadata as base)
        assembled_metadata = first_metadata.copy()
        # Remove micro-batch specific fields
        assembled_metadata.pop("is_microbatch_push", None)
        assembled_metadata.pop("micro_batch_idx", None)
        assembled_metadata.pop("micro_batch_offset", None)
        assembled_metadata.pop("micro_batch_size", None)
        assembled_metadata.pop("full_batch_size", None)
        
        logger.info(
            f"{MBPIPE_LOG_PREFIX} Assembled {len(sorted_mb_indices)} micro-batches: "
            f"{[t.size for t in assembled_tensors if hasattr(t, 'size')]}"
        )
        
        # Create assembled request
        return runtime_pb2.ExpertRequest(
            uid=CHAIN_DELIMITER.join(requested_uids),
            tensors=assembled_tensors,
            metadata=MSGPackSerializer.dumps(assembled_metadata),
        )

    async def _push_outputs(
        self, request: runtime_pb2.ExpertRequest, serialized_outputs: runtime_pb2.Tensor, metadata: dict
    ) -> None:
        # print('_push_outputs metadata ', metadata)
        push_start_time = perf_counter()
        try:
            next_servers = metadata.get("next_servers")
            if not next_servers:
                print(f"[DEBUG] _push_outputs: No next_servers, returning early")
                return

            next_peer_id, next_session_id, next_start, next_end = next_servers[0]
            next_peer_id = PeerID.from_base58(next_peer_id)
            next_uid = CHAIN_DELIMITER.join(f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(next_start, next_end))

            # Log cross-GPU transfer start
            logger.info(f"[CROSS_GPU_TRANSFER_START] FromBlocks={self.dht_prefix} ToBlocks={next_start}:{next_end} ToPeer={next_peer_id}")

            # Sending hidden states serialized with output_schema to avoid double serialization
            next_tensors = [serialized_outputs] + request.tensors[2:]
            next_metadata = metadata.copy()
            next_metadata.update(session_id=next_session_id, next_servers=next_servers[2:], pushed=True)

            stub = self.get_stub(self._p2p, next_peer_id)
            transfer_start = perf_counter()
            await stub.rpc_push(
                runtime_pb2.ExpertRequest(
                    uid=next_uid,
                    tensors=next_tensors,
                    metadata=MSGPackSerializer.dumps(next_metadata),
                ),
                timeout=self.request_timeout,
            )
        except Exception:
            logger.debug(
                f"Failed to push outputs to peer_id={next_peer_id}, session_id={next_session_id}, blocks={next_start}:{next_end}:",
                exc_info=True,
            )

    async def _push_microbatch(
        self,
        mb_hidden: torch.Tensor,
        mb_keep_indices: Optional[torch.Tensor],
        metadata: dict,
        requested_backends: Sequence[TransformerBackend],
    ) -> None:
        """
        [MBPIPE] Push a single micro-batch to the next server for cross-stage overlap.
        
        This enables pipeline parallelism where Server2 can start processing micro-batch N
        while Server1 is still computing micro-batch N+1.
        
        Args:
            mb_hidden: Hidden states tensor for this micro-batch
            mb_keep_indices: Keep indices tensor (for speculative decoding)
            metadata: Contains next_servers, micro_batch_idx, etc.
            requested_backends: Backends for serialization schema
        """
        import os
        # [MBPIPE] Feature flag for cross-stage micro-batch push
        # Default: enabled ("1") since Step 4.2 added Server2 support for receiving micro-batches
        # Set BLOOMBEE_ENABLE_CROSS_STAGE_PUSH=0 to disable
        enable_actual_push = os.environ.get("BLOOMBEE_ENABLE_CROSS_STAGE_PUSH", "1") == "1"
        
        push_start_time = perf_counter()
        
        try:
            next_servers = metadata.get("next_servers")
            if not next_servers:
                return
            
            mb_idx = metadata.get("micro_batch_idx", 0)
            mb_offset = metadata.get("micro_batch_offset", 0)
            mb_size = metadata.get("micro_batch_size", mb_hidden.shape[0])
            full_batch_size = metadata.get("full_batch_size", mb_size)
            
            next_peer_id, next_session_id, next_start, next_end = next_servers[0]
            
            # Log the push intent
            logger.info(
                f"{MBPIPE_LOG_PREFIX} Cross-stage push: mb_idx={mb_idx}, "
                f"offset={mb_offset}, size={mb_size}, to={next_start}:{next_end}"
                f"{'' if enable_actual_push else ' (dry-run, set BLOOMBEE_ENABLE_CROSS_STAGE_PUSH=1 to enable)'}"
            )
            
            # Only actually send if the feature is enabled
            if not enable_actual_push:
                return
            
            next_peer_id = PeerID.from_base58(next_peer_id)
            next_uid = CHAIN_DELIMITER.join(f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(next_start, next_end))
            
            # Serialize the micro-batch hidden states
            outputs_schema = tuple(nested_flatten(requested_backends[-1].outputs_schema))
            
            serialized_hidden = serialize_torch_tensor(
                mb_hidden.to(outputs_schema[0].dtype),
                outputs_schema[0].compression,
                allow_inplace=True
            )
            
            # Serialize keep_indices if present
            if mb_keep_indices is not None:
                serialized_keep = serialize_torch_tensor(
                    mb_keep_indices.to(torch.int64),
                    outputs_schema[1].compression if len(outputs_schema) > 1 else runtime_pb2.CompressionType.NONE,
                    allow_inplace=True
                )
            else:
                serialized_keep = serialize_torch_tensor(
                    torch.arange(mb_hidden.shape[1], dtype=torch.int64),
                    runtime_pb2.CompressionType.NONE,
                    allow_inplace=True
                )
            
            # Build metadata for micro-batch push
            push_metadata = {
                "session_id": next_session_id,
                "next_servers": next_servers[2:] if len(next_servers) > 2 else [],
                "pushed": True,
                # [MBPIPE] Micro-batch specific fields
                "is_microbatch_push": True,
                "micro_batch_idx": mb_idx,
                "micro_batch_offset": mb_offset,
                "micro_batch_size": mb_size,
                "full_batch_size": full_batch_size,
            }
            
            # Copy other relevant metadata
            # [CROSS_STAGE] Include timestamps for cross-stage overlap analysis
            for key in ["step_id", "max_length", "args_structure", "stage_push_timestamp_us", "total_micro_batches", "stage_compute_start_timestamp_us"]:
                if key in metadata:
                    push_metadata[key] = metadata[key]
            
            stub = self.get_stub(self._p2p, next_peer_id)
            
            # [ASYNC_PUSH] Fire-and-forget: don't await RPC response
            # This allows Stage 1 compute to continue immediately while data is sent in background
            rpc_request = runtime_pb2.ExpertRequest(
                uid=next_uid,
                tensors=[serialized_hidden, serialized_keep],
                metadata=MSGPackSerializer.dumps(push_metadata),
            )
            
            # [PHASE2] Flow control: limit concurrent pending pushes to prevent queue buildup
            # Initialize semaphore lazily (max 4 concurrent pushes)
            if not hasattr(self, '_push_semaphore'):
                self._push_semaphore = asyncio.Semaphore(4)
            
            # Acquire semaphore before queueing (non-blocking check for logging)
            sem_wait_start = perf_counter()
            await self._push_semaphore.acquire()
            sem_wait_time = (perf_counter() - sem_wait_start) * 1000
            if sem_wait_time > 1.0:  # Only log if we had to wait
                logger.info(f"{MBPIPE_LOG_PREFIX} [FLOW_CONTROL] MB{mb_idx} waited {sem_wait_time:.1f}ms for push slot")
            
            # Create task for background sending - don't await
            send_task = asyncio.create_task(
                self._do_rpc_push_async(stub, rpc_request, mb_idx, push_start_time)
            )
            
            # Track task to prevent garbage collection
            if not hasattr(self, '_background_push_tasks'):
                self._background_push_tasks = set()
            self._background_push_tasks.add(send_task)
            send_task.add_done_callback(self._background_push_tasks.discard)
            
            queue_time = (perf_counter() - push_start_time) * 1000
            logger.info(f"{MBPIPE_LOG_PREFIX} Micro-batch push queued in {queue_time:.1f}ms (sending in background)")
            
        except Exception as e:
            logger.warning(
                f"{MBPIPE_LOG_PREFIX} Failed to push micro-batch: {e}",
                exc_info=True
            )

    async def _do_rpc_push_async(
        self,
        stub,
        request: runtime_pb2.ExpertRequest,
        mb_idx: int,
        queue_start_time: float,
    ) -> None:
        """
        [ASYNC_PUSH] Actually perform the RPC push in background.
        
        This runs as a fire-and-forget task, allowing the main compute loop
        to continue without waiting for the network round-trip.
        """
        send_start = perf_counter()
        try:
            await stub.rpc_push(request, timeout=self.request_timeout)
            total_time = (perf_counter() - queue_start_time) * 1000
            send_time = (perf_counter() - send_start) * 1000
            logger.info(
                f"{MBPIPE_LOG_PREFIX} [ASYNC_PUSH] MB{mb_idx} sent: "
                f"send={send_time:.1f}ms, total_from_queue={total_time:.1f}ms"
            )
        except Exception as e:
            logger.warning(
                f"{MBPIPE_LOG_PREFIX} [ASYNC_PUSH] MB{mb_idx} send failed: {e}"
            )
        finally:
            # [PHASE2] Release semaphore to allow next push
            if hasattr(self, '_push_semaphore'):
                self._push_semaphore.release()

    async def rpc_forward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.request_timeout):
            # Start timing for server processing latency
            server_start_time = perf_counter()
            
            # Parse request and prepare backends
            flat_inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_forward", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            args_structure = metadata.get("args_structure")
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward should have number of points as number or None, got {points}"

            # Log server processing start
            logger.info(f"[SERVER_PROCESSING_START] Server processing request with {len(requested_uids)} backends")
            
            # Measure network transfer time for S1->S2 communication
            network_start_time = perf_counter()
            hidden_states = await run_rpc_forward(
                *flat_inputs,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
                args_structure=args_structure,
            )
            network_end_time = perf_counter()
            network_transfer_time = (network_end_time - network_start_time) * 1000
            
            # Calculate server processing latency
            server_end_time = perf_counter()
            server_processing_latency = (server_end_time - server_start_time) * 1000
            
            logger.info(f"[NETWORK_TRANSFER_LATENCY] S1->S2 Transfer: {network_transfer_time:.2f}ms | "
                       f"Backends: {len(requested_backends)} | "
                       f"Output Shape: {hidden_states.shape}")
            logger.info(f"[SERVER_PROCESSING_LATENCY] Total: {server_processing_latency:.2f}ms | "
                       f"Backends: {len(requested_backends)} | "
                       f"Output Shape: {hidden_states.shape}")
            
            return runtime_pb2.ExpertResponse(
                tensors=self._serialize_outputs(hidden_states, requested_backends, metadata)
            )

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        async with timeout(self.request_timeout):
            # Parse requests and prepare backends
            uid_str, flat_inputs, metadata = await self._gather_inputs(requests, context)
            requested_uids = self._check_uids(uid_str)
            self._log_request("rpc_forward_stream", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            args_structure = metadata.get("args_structure")
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward_stream should have number of points as number or None, got {points}"

            hidden_states = await run_rpc_forward(
                *flat_inputs,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
                args_structure=args_structure,
            )

            # Split the serialized_output for streaming and respond to client
            for tensor in self._serialize_outputs(hidden_states, requested_backends, metadata):
                for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part])

    def _serialize_outputs(
        self,
        hidden_states: torch.Tensor,
        requested_backends: Sequence[TransformerBackend],
        metadata: Dict[str, Any],
    ) -> Sequence[runtime_pb2.Tensor]:
        """Serialize forward outputs using either outputs_schema or custom user-specified schema"""
        assert isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3, "hidden_states must be a 3d tensor"
        outputs_schema = requested_backends[-1].outputs_schema

        if metadata.get("output_compression") is not None:
            assert isinstance(metadata["output_compression"], (list, tuple)), "output_compression must be a tuple/list"
            output_compression = tuple(metadata["output_compression"])
            assert all(isinstance(c, int) for c in output_compression), "output_compression must contain integers"
            assert len(output_compression) == 1, f"output_compression tuple should have 1 element"
        else:
            output_compression = tuple(tensor.compression for tensor in outputs_schema)

        return [
            serialize_torch_tensor(result.to(proto.dtype), compression, allow_inplace=True)
            for result, proto, compression in zip([hidden_states], outputs_schema, output_compression)
        ]

    async def rpc_backward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.request_timeout):
            # Parse requests and prepare backends
            flat_tensors = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_backward", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            args_structure = metadata.get("args_structure")
            assert isinstance(
                points, (float, int)
            ), f"rpc_backward should have number of points as number or None, got {points}"

            grads = await run_rpc_backward(
                *flat_tensors,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
                args_structure=args_structure,
            )

            return runtime_pb2.ExpertResponse(tensors=self._serialize_grads(grads, requested_backends, metadata))

    async def rpc_backward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        async with timeout(self.request_timeout):
            uids_header, flat_tensors, metadata = await self._gather_inputs(requests, context)
            requested_uids = self._check_uids(uids_header)
            self._log_request("rpc_backward_stream", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            active_adapter = self._get_active_adapter(metadata)
            points = metadata.get("points", 0)
            args_structure = metadata.get("args_structure")
            assert isinstance(
                points, (float, int)
            ), f"rpc_backward_stream should have number of points as number or None, got {points}"

            grads = await run_rpc_backward(
                *flat_tensors,
                requested_backends=requested_backends,
                prioritizer=self._prioritizer,
                active_adapter=active_adapter,
                points=points,
                args_structure=args_structure,
            )
            # Split the serialized_grad_inputs for streaming and respond
            for tensor in self._serialize_grads(grads, requested_backends, metadata):
                for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part])

    def _get_active_adapter(self, metadata: dict) -> str:
        active_adapter = metadata.get("active_adapter", "")
        if active_adapter and (active_adapter not in self.adapters):
            raise KeyError(f"adapter {active_adapter} not found")
        return active_adapter

    def _serialize_grads(
        self,
        grads: Sequence[torch.Tensor],
        requested_backends: Sequence[TransformerBackend],
        metadata: Dict[str, Any],
    ) -> Sequence[runtime_pb2.Tensor]:
        """Serialize backward gradients w.r.t. inputs using either default schema or custom user-specified schema"""
        # Modify grad_inputs_schema to support grad_prompts
        assert len(requested_backends[0].args_schema) == 1 and len(grads) in (1, 2)  # TODO generalize
        flat_grads_schema = tuple(
            nested_flatten((requested_backends[0].args_schema * len(grads), requested_backends[0].kwargs_schema))
        )  # TODO generalize

        if metadata.get("output_compression") is not None:
            assert isinstance(metadata["output_compression"], (list, tuple)), "output_compression must be a tuple/list"
            output_compression = tuple(metadata["output_compression"])
            assert all(isinstance(c, int) for c in output_compression), "output_compression must contain integers"
            assert len(output_compression) == len(grads), f"output_compression should have {len(grads)} elements"
        else:
            output_compression = tuple(tensor.compression for tensor in flat_grads_schema)

        return [
            serialize_torch_tensor(result.to(proto.dtype), compression, allow_inplace=True)
            for result, proto, compression in zip(grads, flat_grads_schema, output_compression)
        ]

    def _check_uids(self, uids: str) -> Tuple[ModuleUID, ...]:
        """Check that the first request to rpc_inference is valid"""
        uids = (uids or "").split(CHAIN_DELIMITER)
        if not uids:
            raise RuntimeError("User did not provide any uids")
        for uid in uids:
            if uid not in self.module_backends:
                raise RuntimeError(f"Remote peer does not serve {uid}")
        return tuple(uids)

    @contextlib.asynccontextmanager
    async def _allocate_cache(
        self,
        backends: Sequence[TransformerBackend],
        *,
        batch_size: int,
        max_length: int,
        timeout: Optional[float],
    ) -> Sequence[Sequence[Handle]]:
        """
        Allocate memory cache for all transformer blocks, return cache handle
        :returns: a list of {len(backends)} elements, where i-th element is a tuple of cache handles for i-th backend
        """
        # offload_logger.info(f" Allocating cache:")
        # offload_logger.info(f"   - Number of backends: {len(backends)}")
        # offload_logger.info(f"   - Batch size: {batch_size}")
        # offload_logger.info(f"   - Max length: {max_length}")
        # offload_logger.info(f"   - Timeout: {timeout}")
        
        # Use KVCacheManager's offloading strategy
        cache_manager = backends[0].cache_manager

        # [TRUE MICRO-BATCH MULTIPLEXING] 
        # When micro-batching is enabled:
        # - GPU cache is sized for micro_batch_size only
        # - Client can request larger batch_size (handled via offload/prefetch)
        # - Each micro-batch reuses the same GPU cache slots
        from bloombee.utils.microbatch_config import get_micro_batch_size, get_micro_batch_config
        from bloombee.utils.memory_usage import log_mbpipe_memory, log_kv_cache_allocation, MemoryTracker
        
        mb_config = get_micro_batch_config()
        policy = cache_manager.offloading_policy
        max_supported_batch = policy.gpu_batch_size
        micro_batch_size = mb_config['micro_batch_size']
        
        # [MBPIPE_DEBUG] Log the critical allocation decision
        logger.info(f"[MBPIPE_ALLOC_DEBUG] ========================================")
        logger.info(f"[MBPIPE_ALLOC_DEBUG] KV CACHE ALLOCATION DECISION POINT")
        logger.info(f"[MBPIPE_ALLOC_DEBUG] ========================================")
        logger.info(f"[MBPIPE_ALLOC_DEBUG] Input: batch_size={batch_size}, max_length={max_length}")
        logger.info(f"[MBPIPE_ALLOC_DEBUG] Config: mb_enabled={mb_config['enabled']}, micro_batch_size={micro_batch_size}")
        logger.info(f"[MBPIPE_ALLOC_DEBUG] Policy: gpu_batch_size={max_supported_batch}")
        
        if mb_config['enabled'] and micro_batch_size < batch_size:
            # [MBPIPE_FIX] GPU Memory Multiplexing:
            # - Allocate GPU cache for micro_batch_size ONLY (not full batch)
            # - All micro-batches reuse the same GPU cache slots (offset=0)
            # - offload() saves current micro-batch's KV to CPU after compute
            # - prefetch() loads next micro-batch's KV from CPU before compute
            # This achieves true GPU memory savings!
            alloc_batch_size = micro_batch_size  # FIX: allocate for micro-batch only!
            
            logger.info(f"[MBPIPE_ALLOC_DEBUG] !!! GPU MEMORY MULTIPLEXING ENABLED !!!")
            logger.info(f"[MBPIPE_ALLOC_DEBUG] alloc_batch_size = {alloc_batch_size} (MICRO-BATCH)")
            logger.info(f"[MBPIPE_ALLOC_DEBUG] Full batch ({batch_size}) will be processed in {(batch_size + micro_batch_size - 1) // micro_batch_size} micro-batches")
            logger.info(f"[MBPIPE_ALLOC_DEBUG] Each micro-batch reuses the same GPU cache slots")
            
            # [MBPIPE_DEBUG] Calculate and log expected memory usage
            try:
                block_config = cache_manager.block_config
                log_kv_cache_allocation(
                    batch_size=batch_size,
                    micro_batch_size=micro_batch_size,
                    max_length=max_length,
                    num_blocks=len(backends),
                    hidden_size=getattr(block_config, 'hidden_size', 4096),
                    num_heads=getattr(block_config, 'num_attention_heads', 32),
                    dtype_bytes=2  # fp16
                )
            except Exception as e:
                logger.debug(f"[MBPIPE_ALLOC_DEBUG] log_kv_cache_allocation failed: {e}")
            
        else:
            # Micro-batching disabled: enforce strict batch limit
            alloc_batch_size = batch_size
            logger.info(f"[MBPIPE_ALLOC_DEBUG] Micro-batching disabled, alloc_batch_size={alloc_batch_size}")
            if batch_size > max_supported_batch:
                raise AllocationFailed(
                    f"Requested batch size {batch_size} exceeds server capacity "
                    f"{max_supported_batch}. Reduce client batch size or restart the "
                    f"server with a larger --batch_size value."
                )
        
        logger.info(f"[MBPIPE_ALLOC_DEBUG] ========================================")
        
        # [MBPIPE_DEBUG] Call the memory savings diagnosis to explain current behavior
        try:
            from bloombee.utils.microbatch_config import log_memory_savings_diagnosis
            log_memory_savings_diagnosis(logger, batch_size)
        except Exception as e:
            logger.debug(f"[MBPIPE_ALLOC_DEBUG] log_memory_savings_diagnosis failed: {e}")

        # Allocate cache descriptors for alloc_batch_size (= micro_batch_size when MB enabled)
        descriptors = [backend.get_inference_cache_descriptors(alloc_batch_size, max_length) for backend in backends]

        logger.info(
            f"OFFLOAD: requesting KV allocation for {len(backends)} blocks, "
            f"alloc_batch={alloc_batch_size}, client_batch={batch_size}, max_length={max_length}"
        )
        
        async with backends[0].cache_manager.allocate_cache(*chain(*descriptors), timeout=timeout) as raw_handles:
            
            logger.info("OFFLOAD: allocation completed; entering use_cache region")
            yield nested_pack(raw_handles, descriptors)

    def _cleanup_warmup_shared_memory(self):
        """
        Clean up temporary shared memory after warmup/prefill phase.
        This helps reduce /dev/shm peak usage on systems with limited shared memory.
        For larger batch sizes, this is called more frequently to prevent accumulation.
        """
        try:
            import gc
            # Force garbage collection to free up temporary objects
            # This helps release shared memory used by temporary Python objects
            gc.collect()
            
            # Clear CUDA cache if available (this may free some shared memory)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()
            
            logger.debug("Cleaned up temporary shared memory after warmup phase")
        except Exception as e:
            logger.debug(f"Failed to cleanup warmup shared memory: {e}", exc_info=True)

    def _log_request(
        self,
        method: str,
        uids: Optional[Sequence[ModuleUID]],
        context: P2PContext,
        *,
        debug: Optional[str] = None,
        warning: Optional[str] = None,
    ) -> None:
        if uids is not None:
            friendly_uids = [uid.split(".")[-1] for uid in uids if "." in uid]
            friendly_uids = [int(uid) for uid in friendly_uids if uid.isdigit()]
            friendly_uids = f"{min(friendly_uids)}:{max(friendly_uids) + 1}" if friendly_uids else uids
        else:
            friendly_uids = "n/a"

        friendly_remote_id = "..." + str(context.remote_id)[-6:]

        message = f"{method}(blocks={friendly_uids}, remote_peer={friendly_remote_id})"
        if warning is not None:
            logger.warning(f"{message}: {warning}")
        elif debug is not None:
            logger.debug(f"{message}: {debug}")
        else:
            logger.info(message)

    async def rpc_info(self, request: runtime_pb2.ExpertUID, context: P2PContext) -> runtime_pb2.ExpertInfo:
        """Return metadata about stored block uids and current load"""

        backend = self.module_backends[request.uid] if request.uid else next(iter(self.module_backends.values()))
        result = {
            "version": bloombee.__version__,
            "dht_client_mode": self.dht.client_mode,
            CACHE_TOKENS_AVAILABLE: backend.cache_manager.tokens_left(),
        }

        if request.uid:
            block_info = self.module_backends[request.uid].get_info()
            common_keys = set(result.keys()) & set(block_info.keys())
            if common_keys:
                raise RuntimeError(f"The block's rpc_info has keys reserved for the server's rpc_info: {common_keys}")
            result.update(block_info)

        return runtime_pb2.ExpertInfo(serialized_info=MSGPackSerializer.dumps(result))
