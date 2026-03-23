from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import sys
from collections import deque
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Any, AsyncIterator, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from time import perf_counter
import time
import numpy as np

import torch
from async_timeout import timeout
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
from bloombee.server.microbatch import resolve_expected_num_microbatches
from bloombee.server.task_prioritizer import DummyTaskPrioritizer, TaskPrioritizerBase
from bloombee.utils.hivemind_compat import DHT, MSGPackSerializer, P2PContext, PeerID, nested_flatten, nested_pack
from bloombee.utils.convert_block import QuantType
from bloombee.utils.debug_config import is_log_channel_enabled
from bloombee.utils.lossless_transport import (
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    serialize_torch_tensor,
    log_comp_ratio_event,
    log_transport_profile_event,
    transport_profile_scope,
    tensor_nnz_ratio,
    tensor_raw_nbytes,
)
from bloombee.utils.packaging import unpack_args_kwargs
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
    create_microbatch_queue_item,
    is_microbatch_queue_item,
    MBPIPE_SCHEMA_PREFIX,
)

logger = get_logger(__name__)

_s2s_output_compression_name = os.getenv("BLOOMBEE_S2S_OUTPUT_COMPRESSION", "").strip().upper()
_s2s_output_compression = None
if _s2s_output_compression_name:
    try:
        _s2s_output_compression = getattr(runtime_pb2.CompressionType, _s2s_output_compression_name)
    except AttributeError:
        logger.warning(
            "Unknown BLOOMBEE_S2S_OUTPUT_COMPRESSION=%r, falling back to default rpc_push compression",
            _s2s_output_compression_name,
        )

if TYPE_CHECKING:
    from bloombee.server.speculative_pruner.pruner_manager import SpeculativePrunerManager


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
@dataclass
class S2SLinkTelemetry:
    """
    Rolling transport telemetry for one server-to-server link.

    This is used to distinguish real throughput changes from network variance:
    we log latency, bandwidth and jitter over a sliding window so experiments
    can show whether the network stayed stable while throughput changed.
    """

    label: str
    window_size: int
    samples: int = 0
    total_bytes: int = 0
    clock_sync_samples: int = 0
    last_latency_ms: Optional[float] = None
    latency_ms_window: Deque[float] = field(init=False)
    bandwidth_mbps_window: Deque[float] = field(init=False)
    jitter_ms_window: Deque[float] = field(init=False)
    raw_latency_ms_window: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.latency_ms_window = deque(maxlen=self.window_size)
        self.bandwidth_mbps_window = deque(maxlen=self.window_size)
        self.jitter_ms_window = deque(maxlen=self.window_size)
        self.raw_latency_ms_window = deque(maxlen=self.window_size)

    def record(
        self,
        *,
        latency_ms: float,
        raw_latency_ms: float,
        bandwidth_mbps: float,
        total_bytes: int,
        clock_sync_ok: bool,
    ) -> float:
        jitter_ms = 0.0
        if self.last_latency_ms is not None:
            jitter_ms = abs(float(latency_ms) - float(self.last_latency_ms))
        self.last_latency_ms = float(latency_ms)

        self.samples += 1
        self.total_bytes += max(0, int(total_bytes))
        if clock_sync_ok:
            self.clock_sync_samples += 1
        self.latency_ms_window.append(float(latency_ms))
        self.bandwidth_mbps_window.append(float(bandwidth_mbps))
        self.jitter_ms_window.append(float(jitter_ms))
        self.raw_latency_ms_window.append(float(raw_latency_ms))
        return jitter_ms

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


class AdaptivePushConcurrency:
    """
    Self-tuning limiter for cross-stage micro-batch pushes.

    The limiter adjusts in-flight push concurrency from runtime signals:
    - acquire wait (queue pressure inside sender)
    - RPC send duration (network pressure)
    - send failures (stability signal)

    No external tuning knobs are required; limits are bounded to keep behavior stable.
    """

    def __init__(
        self,
        *,
        logger_: logging.Logger,
        name: str,
        initial_limit: int = 4,
        min_limit: int = 2,
        max_limit: int = 12,
        ewma_alpha: float = 0.2,
        decision_interval: int = 8,
    ):
        self._logger = logger_
        self._name = name
        self._limit = max(min_limit, min(max_limit, int(initial_limit)))
        self._min_limit = int(min_limit)
        self._max_limit = int(max_limit)
        self._ewma_alpha = float(ewma_alpha)
        self._decision_interval = max(1, int(decision_interval))

        self._in_flight = 0
        self._cond = asyncio.Condition()

        self._ewma_wait_ms = 0.0
        self._ewma_send_ms = 0.0
        self._release_events = 0
        self._recent_failures = 0
        self._consecutive_failures = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def _update_ewma(self, prev: float, sample: float) -> float:
        if prev <= 0.0:
            return sample
        a = self._ewma_alpha
        return prev * (1.0 - a) + sample * a

    async def acquire(self) -> float:
        wait_start = perf_counter()
        async with self._cond:
            while self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1
        wait_ms = (perf_counter() - wait_start) * 1000.0
        self._ewma_wait_ms = self._update_ewma(self._ewma_wait_ms, wait_ms)
        return wait_ms

    async def release(self, *, send_time_ms: float, success: bool) -> None:
        change_log = None
        async with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            self._ewma_send_ms = self._update_ewma(self._ewma_send_ms, max(0.0, float(send_time_ms)))

            if success:
                self._consecutive_failures = 0
                self._recent_failures = max(0, self._recent_failures - 1)
            else:
                self._consecutive_failures += 1
                self._recent_failures = min(16, self._recent_failures + 1)

            self._release_events += 1
            if self._release_events % self._decision_interval == 0:
                old_limit = self._limit
                reason = None

                # Stability first: back off quickly on repeated failures.
                if self._consecutive_failures >= 2 or self._recent_failures >= 3:
                    self._limit = max(self._min_limit, self._limit - 1)
                    self._consecutive_failures = 0
                    reason = "send_failures"
                # If local wait is non-trivial while network send remains moderate,
                # increase concurrency to reduce sender-side queue pressure.
                elif self._ewma_wait_ms > 8.0 and self._ewma_send_ms < 220.0 and self._in_flight >= max(1, self._limit - 1):
                    self._limit = min(self._max_limit, self._limit + 1)
                    reason = "queue_pressure"
                # If network send slows down a lot, decrease concurrency to avoid congestion collapse.
                elif self._ewma_send_ms > 320.0 and self._ewma_wait_ms < 2.0:
                    self._limit = max(self._min_limit, self._limit - 1)
                    reason = "network_backpressure"

                if self._limit != old_limit:
                    change_log = (
                        old_limit,
                        self._limit,
                        reason or "unspecified",
                        self._ewma_wait_ms,
                        self._ewma_send_ms,
                        self._in_flight,
                    )

            self._cond.notify_all()

        if change_log is not None:
            old_limit, new_limit, reason, ewma_wait_ms, ewma_send_ms, in_flight = change_log
            self._logger.info(
                f"{MBPIPE_LOG_PREFIX} [FLOW_CONTROL] adaptive_limit[{self._name}] "
                f"{old_limit}->{new_limit} reason={reason} "
                f"ewma_wait={ewma_wait_ms:.1f}ms ewma_send={ewma_send_ms:.1f}ms in_flight={in_flight}"
            )


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
        pruner_manager: Optional[SpeculativePrunerManager],
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
        self._session_timing: Dict[str, list] = {}
        self._session_comm_timing: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._session_background_push_tasks: Dict[str, set] = {}
        # [MBPIPE] Cross-stage pipeline: micro-batch queues for immediate processing
        # Key: (session_id, step_id) -> Queue holding individual micro-batches
        self._mb_queues: Dict[tuple, asyncio.Queue] = {}
        # Key: (session_id, step_id) -> expected number of micro-batches
        self._mb_expected: Dict[tuple, int] = {}
        # Key: (session_id, step_id) -> count of received micro-batches
        self._mb_received: Dict[tuple, int] = {}
        # Key: (session_id, step_id) -> set of (mb_idx) already processed (idempotency)
        self._mb_processed: Dict[tuple, set] = {}
        # Feature flag for immediate queuing - ENABLED for cross-stage pipeline overlap
        self._enable_immediate_mb_queue = os.environ.get(
            "BLOOMBEE_ENABLE_IMMEDIATE_MB_QUEUE", "1"
        ) == "1"
        # Optional compatibility mode: also write MB payloads to filesystem in immediate mode.
        # Disabled by default to reduce first-micro-batch latency.
        self._store_mb_files_in_immediate = os.environ.get(
            "BLOOMBEE_STORE_MB_FILES_IN_IMMEDIATE", "0"
        ) == "1"
        
        logger.info(f"{MBPIPE_LOG_PREFIX} Immediate micro-batch queuing: {'ENABLED' if self._enable_immediate_mb_queue else 'disabled'}")
        logger.info(
            f"{MBPIPE_LOG_PREFIX} Immediate-mode file store: "
            f"{'ENABLED' if self._store_mb_files_in_immediate else 'disabled'} "
            f"(set BLOOMBEE_STORE_MB_FILES_IN_IMMEDIATE=1 for legacy behavior)"
        )
        
        # [CLOCK_SYNC] Per-peer clock offset estimator for cross-machine strict overlap.
        # offset_us is "remote_clock - local_clock" for the target peer.
        self._clock_sync_state: Dict[str, Dict[str, float]] = {}
        self._clock_sync_alpha = float(os.environ.get("BLOOMBEE_CLOCK_SYNC_ALPHA", "0.2"))
        self._clock_sync_max_rtt_us = max(0, int(os.environ.get("BLOOMBEE_CLOCK_SYNC_MAX_RTT_US", "2000000")))
        self._clock_sync_log_every = max(1, int(os.environ.get("BLOOMBEE_CLOCK_SYNC_LOG_EVERY", "64")))
        logger.info(
            f"{MBPIPE_LOG_PREFIX} Clock sync enabled: alpha={self._clock_sync_alpha:.2f}, "
            f"max_rtt={self._clock_sync_max_rtt_us/1000:.1f}ms, log_every={self._clock_sync_log_every}"
        )

        # [S2S_TELEMETRY] Rolling link telemetry for server-to-server transport.
        # This makes it easier to verify that throughput changes are not caused by
        # transient network jitter or bandwidth fluctuations.
        self._s2s_stats_window = max(4, int(os.environ.get("BLOOMBEE_S2S_STATS_WINDOW", "32")))
        self._s2s_stats_log_every = max(1, int(os.environ.get("BLOOMBEE_S2S_STATS_LOG_EVERY", "8")))
        self._s2s_link_stats: Dict[str, S2SLinkTelemetry] = {}
        logger.info(
            f"{MBPIPE_LOG_PREFIX} S2S telemetry enabled: "
            f"window={self._s2s_stats_window}, log_every={self._s2s_stats_log_every}"
        )

        # [FLOW_CONTROL] Internal adaptive limiter for cross-stage async pushes.
        # Keeps pipeline stable while seeking higher throughput from runtime feedback.
        self._push_limiter = AdaptivePushConcurrency(
            logger_=logger,
            name=self.dht_prefix,
            initial_limit=4,
            min_limit=2,
            max_limit=12,
        )



        self.inference_max_length = inference_max_length
        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout
        self._prioritizer = task_prioritizer
        self.quant_type = quant_type
        self.pruner_manager = pruner_manager
        self._speculative_pruner_enabled = pruner_manager is not None

    @staticmethod
    def _now_us() -> int:
        return int(time.time() * 1_000_000)

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _get_clock_sync_estimate(self, peer_id: str) -> Optional[Dict[str, int]]:
        state = self._clock_sync_state.get(peer_id)
        if not state:
            return None
        return {
            "offset_us": int(round(float(state.get("offset_us", 0.0)))),
            "rtt_us": int(round(float(state.get("rtt_us", 0.0)))),
            "samples": int(state.get("samples", 0)),
        }

    def _update_clock_sync_estimate(self, peer_id: str, sample_offset_us: float, sample_rtt_us: float) -> Optional[Dict[str, float]]:
        """
        Update per-peer clock offset estimate.
        sample_offset_us follows NTP convention: remote_clock - local_clock.
        """
        if sample_rtt_us < 0:
            return None
        if self._clock_sync_max_rtt_us > 0 and sample_rtt_us > self._clock_sync_max_rtt_us:
            return None

        sample_offset_us = float(sample_offset_us)
        sample_rtt_us = float(sample_rtt_us)
        state = self._clock_sync_state.get(peer_id)
        if state is None:
            state = {
                "offset_us": sample_offset_us,
                "rtt_us": sample_rtt_us,
                "best_rtt_us": sample_rtt_us,
                "samples": 1,
            }
        else:
            prev_offset_us = float(state.get("offset_us", sample_offset_us))
            prev_rtt_us = float(state.get("rtt_us", sample_rtt_us))
            best_rtt_us = min(float(state.get("best_rtt_us", sample_rtt_us)), sample_rtt_us)
            # Lower RTT samples are usually more reliable for offset estimation.
            quality = best_rtt_us / max(sample_rtt_us, 1.0)
            effective_alpha = min(1.0, max(0.01, self._clock_sync_alpha * quality))
            state["offset_us"] = prev_offset_us * (1.0 - effective_alpha) + sample_offset_us * effective_alpha
            state["rtt_us"] = prev_rtt_us * (1.0 - effective_alpha) + sample_rtt_us * effective_alpha
            state["best_rtt_us"] = best_rtt_us
            state["samples"] = int(state.get("samples", 0)) + 1

        state["last_raw_offset_us"] = sample_offset_us
        state["last_raw_rtt_us"] = sample_rtt_us
        state["updated_at_us"] = self._now_us()
        self._clock_sync_state[peer_id] = state
        return state

    def _update_clock_sync_from_rpc_response(
        self,
        peer_id: str,
        sender_send_us: int,
        sender_ack_us: int,
        response: Optional[runtime_pb2.ExpertResponse],
    ) -> None:
        if response is None or not response.metadata:
            return
        try:
            response_meta = MSGPackSerializer.loads(response.metadata)
        except Exception:
            return
        if not isinstance(response_meta, dict):
            return

        receiver_recv_us = self._to_int(response_meta.get("clock_sync_receiver_recv_us"), 0)
        receiver_ack_us = self._to_int(response_meta.get("clock_sync_receiver_ack_us"), 0)
        if receiver_recv_us <= 0 or receiver_ack_us <= 0 or sender_ack_us < sender_send_us:
            return

        # NTP four-timestamp estimator:
        # t1=sender_send, t2=receiver_recv, t3=receiver_ack, t4=sender_ack.
        # offset (receiver-local) = ((t2-t1) + (t3-t4))/2
        receiver_processing_us = max(0, receiver_ack_us - receiver_recv_us)
        end_to_end_rtt_us = max(0, sender_ack_us - sender_send_us)
        network_rtt_us = max(0, end_to_end_rtt_us - receiver_processing_us)
        sample_offset_us = ((receiver_recv_us - sender_send_us) + (receiver_ack_us - sender_ack_us)) / 2.0

        updated = self._update_clock_sync_estimate(peer_id, sample_offset_us, network_rtt_us)
        if not updated:
            return
        samples = int(updated.get("samples", 0))
        if samples <= 3 or (samples % self._clock_sync_log_every == 0):
            logger.info(
                f"{MBPIPE_LOG_PREFIX} [CLOCK_SYNC] peer={peer_id[:10]} "
                f"offset={updated['offset_us']/1000:.2f}ms "
                f"rtt={updated['rtt_us']/1000:.2f}ms samples={samples}"
            )

    def _build_rpc_push_ack_response(self, receive_us: int) -> runtime_pb2.ExpertResponse:
        ack_metadata = {
            "clock_sync_receiver_recv_us": int(receive_us),
            "clock_sync_receiver_ack_us": int(self._now_us()),
        }
        return runtime_pb2.ExpertResponse(metadata=MSGPackSerializer.dumps(ack_metadata))

    @staticmethod
    def _calc_mbps(total_bytes: int, latency_ms: float) -> float:
        if total_bytes <= 0 or latency_ms <= 0:
            return 0.0
        return (float(total_bytes) * 8.0) / (float(latency_ms) * 1000.0)

    def _record_session_comm_timing(
        self,
        session_id: Optional[str],
        step_id: Optional[str],
        *,
        t_cpu2nic_ms: float,
        t_nic2nic_ms: float,
        push_e2e_ms: float,
        receiver_processing_ms: float,
        wire_bytes: float,
    ) -> None:
        if not session_id or not step_id or step_id == "unknown":
            return
        session_records = self._session_comm_timing.setdefault(session_id, {})
        record = session_records.setdefault(
            step_id,
            {
                "t_cpu2nic_ms": 0.0,
                "t_nic2nic_ms": 0.0,
                "push_e2e_ms": 0.0,
                "receiver_processing_ms": 0.0,
                "wire_bytes": 0.0,
                "samples": 0,
            },
        )
        record["t_cpu2nic_ms"] += float(t_cpu2nic_ms)
        record["t_nic2nic_ms"] += float(t_nic2nic_ms)
        record["push_e2e_ms"] += float(push_e2e_ms)
        record["receiver_processing_ms"] += float(receiver_processing_ms)
        record["wire_bytes"] += float(wire_bytes)
        record["samples"] += 1

    @staticmethod
    def _emit_unconditional_summary(message: str) -> None:
        print(message, flush=True)

    def _track_session_push_task(self, session_id: Optional[str], task: asyncio.Task) -> None:
        if not session_id:
            return
        session_tasks = self._session_background_push_tasks.setdefault(session_id, set())
        session_tasks.add(task)
        task.add_done_callback(session_tasks.discard)

    async def _await_session_push_tasks(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        pending = tuple(self._session_background_push_tasks.pop(session_id, set()))
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _normalize_serialized_tensors(
        tensors: Union[runtime_pb2.Tensor, Sequence[runtime_pb2.Tensor]]
    ) -> List[runtime_pb2.Tensor]:
        normalized = list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]
        if any(not hasattr(tensor, "buffer") for tensor in normalized):
            bad_types = [type(tensor).__name__ for tensor in normalized]
            raise TypeError(f"Expected serialized runtime_pb2.Tensor objects, got {bad_types}")
        return normalized

    @staticmethod
    def _window_stats(values: Sequence[float]) -> Tuple[float, float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        return mean, std, p50, p95

    @staticmethod
    def _classify_link_stability(latency_mean_ms: float, latency_std_ms: float, jitter_p95_ms: float) -> str:
        if latency_mean_ms <= 0:
            return "unknown"
        cv = latency_std_ms / latency_mean_ms
        if cv <= 0.05 and jitter_p95_ms <= max(2.0, latency_mean_ms * 0.10):
            return "stable"
        if cv <= 0.15 and jitter_p95_ms <= max(5.0, latency_mean_ms * 0.25):
            return "moderate"
        return "volatile"

    @staticmethod
    def _uids_to_block_span_label(uids: Union[str, Sequence[str]]) -> str:
        if isinstance(uids, str):
            uid_items = [item for item in uids.split(CHAIN_DELIMITER) if item]
        else:
            uid_items = [str(item) for item in uids if item]
        indices: List[int] = []
        for uid in uid_items:
            try:
                indices.append(int(uid.split(UID_DELIMITER)[-1]))
            except Exception:
                continue
        if not indices:
            return "unknown"
        return f"{min(indices)}:{max(indices) + 1}"

    def _record_s2s_network_sample(
        self,
        *,
        channel: str,
        sender_blocks: str,
        receiver_blocks: str,
        payload_bytes: int,
        metadata_bytes: int,
        raw_transfer_ms: float,
        wire_ms: float,
        clock_sync_ok: bool,
    ) -> None:
        effective_latency_ms = wire_ms if wire_ms > 0 else raw_transfer_ms
        if effective_latency_ms <= 0:
            return

        total_bytes = max(0, int(payload_bytes)) + max(0, int(metadata_bytes))
        bandwidth_mbps = self._calc_mbps(total_bytes, effective_latency_ms)
        link_key = f"{channel}:{sender_blocks}->{receiver_blocks}"
        telemetry = self._s2s_link_stats.get(link_key)
        if telemetry is None:
            telemetry = S2SLinkTelemetry(label=link_key, window_size=self._s2s_stats_window)
            self._s2s_link_stats[link_key] = telemetry

        jitter_ms = telemetry.record(
            latency_ms=effective_latency_ms,
            raw_latency_ms=raw_transfer_ms if raw_transfer_ms > 0 else effective_latency_ms,
            bandwidth_mbps=bandwidth_mbps,
            total_bytes=total_bytes,
            clock_sync_ok=clock_sync_ok,
        )

        logger.info(
            f"[S2S_NET] link={link_key} samples={telemetry.samples} "
            f"latency_ms={effective_latency_ms:.3f} "
            f"bandwidth_mbps={bandwidth_mbps:.3f} "
            f"jitter_ms={jitter_ms:.3f} "
            f"payload_kb={payload_bytes / 1024.0:.2f} "
            f"metadata_b={metadata_bytes} "
            f"clock_sync={int(clock_sync_ok)}"
        )

        if telemetry.samples <= 3 or telemetry.samples % self._s2s_stats_log_every == 0:
            latency_mean_ms, latency_std_ms, latency_p50_ms, latency_p95_ms = self._window_stats(
                list(telemetry.latency_ms_window)
            )
            bw_mean_mbps, bw_std_mbps, bw_p50_mbps, bw_p95_mbps = self._window_stats(
                list(telemetry.bandwidth_mbps_window)
            )
            jitter_mean_ms, jitter_std_ms, jitter_p50_ms, jitter_p95_ms = self._window_stats(
                list(telemetry.jitter_ms_window)
            )
            stability = self._classify_link_stability(latency_mean_ms, latency_std_ms, jitter_p95_ms)
            clock_sync_coverage = (
                100.0 * float(telemetry.clock_sync_samples) / float(telemetry.samples)
                if telemetry.samples > 0
                else 0.0
            )
            logger.info(
                f"[S2S_NET_SUMMARY] link={link_key} window={len(telemetry.latency_ms_window)} "
                f"samples={telemetry.samples} stability={stability} "
                f"lat_mean={latency_mean_ms:.3f}ms lat_std={latency_std_ms:.3f}ms "
                f"lat_p50={latency_p50_ms:.3f}ms lat_p95={latency_p95_ms:.3f}ms "
                f"jit_mean={jitter_mean_ms:.3f}ms jit_std={jitter_std_ms:.3f}ms "
                f"jit_p50={jitter_p50_ms:.3f}ms jit_p95={jitter_p95_ms:.3f}ms "
                f"bw_mean={bw_mean_mbps:.3f}Mbps bw_std={bw_std_mbps:.3f}Mbps "
                f"bw_p50={bw_p50_mbps:.3f}Mbps bw_p95={bw_p95_mbps:.3f}Mbps "
                f"bytes_total_mb={telemetry.total_bytes / (1024.0 * 1024.0):.3f} "
                f"clock_sync_coverage={clock_sync_coverage:.1f}%"
            )


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
                recv_start = perf_counter()
                request = await asyncio.wait_for(anext(requests), self.step_timeout)
                recv_end = perf_counter()
            except asyncio.TimeoutError:
                self._log_request("rpc_inference.open", None, context, warning="timed out")
                return

            # [NETWORK_TIMING] Log received request size and timing
            request_tensor_sizes = [len(tensor.buffer) for tensor in request.tensors]
            request_metadata_size = len(request.metadata) if request.metadata else 0
            total_request_size = sum(request_tensor_sizes) + request_metadata_size
            recv_time_ms = (recv_end - recv_start) * 1000
            
            logger.info(f"[NETWORK_RX] SERVER_RECV | "
                       f"tensor_size={sum(request_tensor_sizes)/1024:.2f}KB | "
                       f"metadata_size={request_metadata_size}B | "
                       f"total={total_request_size/1024:.2f}KB | "
                       f"recv_time={recv_time_ms:.2f}ms")
            

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
                def _flag_to_bool(value: Any) -> bool:
                    if value is None:
                        return False
                    if torch.is_tensor(value):
                        if value.numel() == 0:
                            return False
                        return bool(value.bool().any().item())
                    return bool(value)

                is_spec_request = _flag_to_bool(metadata.get("is_spec_dec", 0))
                if is_spec_request and not self._speculative_pruner_enabled:
                    logger.info(
                        f"{MBPIPE_LOG_PREFIX} Speculative decoding requested without an active pruner; "
                        f"continuing without branch pruning for session_id={session_id}"
                    )
                    metadata["need_pruning"] = False
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
                metadata_full_batch_size = metadata.get("full_batch_size")
                try:
                    metadata_full_batch_size = int(metadata_full_batch_size) if metadata_full_batch_size is not None else None
                except Exception:
                    metadata_full_batch_size = None
                if metadata_full_batch_size is not None and metadata_full_batch_size <= 0:
                    metadata_full_batch_size = None
                
                # [MB_DEBUG] Log initial batch size detection
                logger.debug(f"[MB_DEBUG] === BATCH SIZE DETECTION ===")
                logger.debug(f"[MB_DEBUG] Original batch_size from tensor[0]: {original_batch_size}")
                logger.debug(f"[MB_DEBUG] Metadata keys: {list(metadata.keys())}")
                logger.debug(f"[MB_DEBUG] metadata.type={metadata.get('type')}, metadata.full_batch_size={metadata.get('full_batch_size')}")
                
                # [MBPIPE_STREAMING] For cross-stage micro-batch streaming, use full_batch_size for KV cache allocation
                # This is critical for decode overlap: Stage 2 must allocate cache for full batch on first MB arrival
                is_streaming_decode = is_microbatch_queue_item(request) or metadata.get("type") == "micro_batch"
                logger.debug(f"[MB_DEBUG] is_streaming_decode={is_streaming_decode}, is_microbatch_queue_item={is_microbatch_queue_item(request)}")
                
                if is_streaming_decode:
                    streaming_full_batch_size = metadata_full_batch_size if metadata_full_batch_size is not None else batch_size
                    logger.debug(f"[MB_DEBUG] Streaming decode detected! streaming_full_batch_size={streaming_full_batch_size}")
                    
                    if is_spec_request and streaming_full_batch_size > batch_size:
                        logger.info(
                            f"{MBPIPE_LOG_PREFIX} Spec streaming request: using full_batch_size={streaming_full_batch_size} "
                            f"for KV allocation (actual incoming mb={batch_size})"
                        )
                        batch_size = streaming_full_batch_size
                    # [MBPIPE_FIX] Keep the incoming request batch at micro-batch size for streaming.
                    # The logical full batch is still passed separately to _allocate_cache(),
                    # which decides how many GPU working slots to reserve.
                    elif is_microbatch_enabled() and streaming_full_batch_size > batch_size:
                        logger.info(
                            f"[MBPIPE_FIX] Micro-batch enabled: keeping request_batch={batch_size} "
                            f"(incoming micro-batch), while logical_full_batch={streaming_full_batch_size} "
                            f"will drive working-slot allocation in _allocate_cache"
                        )
                    elif streaming_full_batch_size > batch_size:
                        logger.info(f"{MBPIPE_LOG_PREFIX} [STREAMING_DECODE] Detected streaming micro-batch (LEGACY), "
                                    f"using full_batch_size={streaming_full_batch_size} for KV cache (actual MB size={batch_size})")
                        batch_size = streaming_full_batch_size
                        logger.debug(f"[MB_DEBUG] KV cache will use batch_size={batch_size} (overridden from {original_batch_size})")
                else:
                    if is_spec_request and metadata_full_batch_size is not None and metadata_full_batch_size > batch_size:
                        logger.info(
                            f"{MBPIPE_LOG_PREFIX} Spec request: override batch_size {batch_size} -> "
                            f"full_batch_size {metadata_full_batch_size} for stable KV allocation"
                        )
                        batch_size = metadata_full_batch_size
                    # Non-streaming RPC path keeps logical full batch size here.
                    # Physical KV cache allocation is decided in _allocate_cache():
                    # when micro-batching is enabled and batch_size > micro_batch_size,
                    # we either keep full-batch KV (overlap-only) or allocate up to
                    # (micro_batch_size * working_slots) in explicit multiplexing mode.
                    if is_microbatch_enabled():
                        first_backend = requested_backends[0]
                        offloading_policy = first_backend.cache_manager.offloading_policy
                        micro_batch_size = get_micro_batch_size()
                        working_slots = max(1, int(getattr(offloading_policy, "num_gpu_batches", 1)))
                        if get_current_path() == "multiplexing":
                            logger.info(
                                f"[MBPIPE_FIX] Non-streaming: logical batch_size={batch_size}, "
                                f"physical alloc will use up to {micro_batch_size * working_slots} "
                                f"(slot_batch={micro_batch_size}, working_slots={working_slots}) in _allocate_cache"
                            )
                        else:
                            logger.info(
                                f"[MBPIPE_FIX] Non-streaming overlap-only mode: logical batch_size={batch_size}, "
                                f"KV allocation stays full-batch in _allocate_cache; "
                                f"micro_batch_size={micro_batch_size}, working_slots={working_slots}"
                            )
                    else:
                        logger.debug(f"[MB_DEBUG] NOT streaming decode, using original batch_size={batch_size}")
                
                logger.debug(f"[MB_DEBUG] Batch size detection completed, final batch_size={batch_size}")
                # print_time_now('')
                
                # [MBPIPE] Log current path at rpc_inference entry
                mbpipe_log_path_entry(logger, "handler.rpc_inference", batch_size=batch_size)
                if is_spec_request:
                    logger.info(
                        f"{MBPIPE_LOG_PREFIX} Speculative decoding request detected; "
                        f"forcing full-batch KV allocation for this session"
                    )
                
                # [MBPIPE] Log comprehensive runtime info
                from bloombee.utils.microbatch_config import log_microbatch_runtime_info
                log_microbatch_runtime_info(
                    logger,
                    batch_size=batch_size,
                    seq_len=max_length,
                    num_blocks=len(requested_backends),
                    context="rpc_inference entry"
                )

                
                push_time = []
                background_tasks = set()
                
                # [KVCACHE_DEBUG] Log before cache allocation
                cache_alloc_start = perf_counter()
                logger.debug(f"[KVCACHE_DEBUG] === KV CACHE ALLOCATION ===")
                logger.debug(f"[KVCACHE_DEBUG] Allocating cache: batch_size={batch_size}, max_length={max_length}, timeout={alloc_timeout}")
                logger.debug(f"[KVCACHE_DEBUG] Requested backends: {len(requested_backends)}, UIDs: {requested_uids}")
                
                async with self._allocate_cache(
                    requested_backends,
                    batch_size=batch_size,
                    logical_full_batch_size=metadata_full_batch_size,
                    max_length=max_length,
                    timeout=alloc_timeout,
                    force_full_batch_alloc=is_spec_request,
                ) as cache_handles:
                    end_cache_time = perf_counter()
                    cache_alloc_ms = (end_cache_time - cache_alloc_start) * 1000
                    
                    # [KVCACHE_DEBUG] Log cache allocation result
                    logger.debug(f"[KVCACHE_DEBUG] Cache allocated in {cache_alloc_ms:.2f}ms")
                    logger.debug(f"[KVCACHE_DEBUG] cache_handles count: {len(cache_handles) if cache_handles else 0}")
                    if cache_handles:
                        for i, handles in enumerate(cache_handles):
                            logger.debug(f"[KVCACHE_DEBUG] cache_handles[{i}]: {len(handles) if handles else 0} handles")

                    background_tasks = set()
                    background_task_errors: List[Exception] = []

                    def _track_background_task(task: asyncio.Task) -> None:
                        """Track task lifecycle and capture async push failures."""
                        background_tasks.add(task)

                        def _on_done(done_task: asyncio.Task) -> None:
                            background_tasks.discard(done_task)
                            if done_task.cancelled():
                                return
                            exc = done_task.exception()
                            if exc is not None:
                                background_task_errors.append(exc)
                                logger.warning(
                                    f"{MBPIPE_LOG_PREFIX} Async push task failed: {exc}",
                                    exc_info=True,
                                )

                        task.add_done_callback(_on_done)

                    async def _drain_background_tasks() -> None:
                        """Wait for pending background pushes and surface failures."""
                        if not background_tasks:
                            return
                        pending = list(background_tasks)
                        results = await asyncio.gather(*pending, return_exceptions=True)
                        for result in results:
                            if isinstance(result, Exception):
                                background_task_errors.append(result)
                        if background_task_errors:
                            raise RuntimeError("Background push tasks failed") from background_task_errors[0]
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
                                await self._push_outputs(req, tensors, meta, raise_on_error=True)
                            
                            await output_buffer.start_sender(buffered_push_fn)
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} AsyncOutputBuffer started for cross-stage overlap"
                            )
                        elif use_buffer and buffer_pos == "consumer":
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} Buffer decision=consumer; "
                                f"consumer-side buffering is not implemented, using direct push path"
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
                        cross_stage_push_fn=cross_stage_push_microbatch,  # [MBPIPE] Cross-stage streaming (currently disabled)
                    ):
                        handler_step_start = perf_counter()
                        step_id_for_log = (
                            step_metadata.get("step_id", "unknown")
                            if isinstance(step_metadata, dict)
                            else "unknown"
                        )
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
                                    task = asyncio.create_task(
                                        self._push_outputs(
                                            request,
                                            output_tensors,
                                            step_metadata,
                                            raise_on_error=True,
                                        )
                                    )
                                    _track_background_task(task)
                            else:
                                # Original direct task creation
                                task = asyncio.create_task(
                                    self._push_outputs(
                                        request,
                                        output_tensors,
                                        step_metadata,
                                        raise_on_error=True,
                                    )
                                )
                                _track_background_task(task)
                        start_ExpertResponse_time=perf_counter() ###
                        push_schedule_ms = (start_ExpertResponse_time - can_push_case_time) * 1000.0
                        push_time.append(push_schedule_ms) ###
                        # print('current step push outputs task prepare time ', start_ExpertResponse_time-can_push_case_time) ###
                        # print_time_now('')
                        yield runtime_pb2.ExpertResponse(tensors=output_tensors)
                        end_ExpertResponse_time=perf_counter() ###
                        response_emit_ms = (end_ExpertResponse_time - start_ExpertResponse_time) * 1000.0
                        handler_step_total_ms = (end_ExpertResponse_time - handler_step_start) * 1000.0
                        queue_wait_ms = (
                            float(step_metadata.get("_queue_wait_ms", 0.0))
                            if isinstance(step_metadata, dict)
                            else 0.0
                        )
                        queue_source = (
                            str(step_metadata.get("_queue_source", "unknown"))
                            if isinstance(step_metadata, dict)
                            else "unknown"
                        )
                        if is_log_channel_enabled("handler_step_timing_logs"):
                            logger.info(
                                f"[HANDLER_STEP_TIMING] step_id={step_id_for_log} "
                                f"queue_wait={queue_wait_ms:.2f}ms queue_source={queue_source} "
                                f"push_schedule={push_schedule_ms:.2f}ms "
                                f"response_emit={response_emit_ms:.2f}ms "
                                f"handler_total={handler_step_total_ms:.2f}ms "
                                f"can_push={int(bool(can_push))}"
                            )
                        if isinstance(step_metadata, dict) and session_id:
                            rec = {
                                "step_id": step_id_for_log,
                                "t_nic2cpu_ms": float(step_metadata.get("_t_nic2cpu_ms", 0)),
                                "t_cpu2gpu_ms": float(step_metadata.get("_t_cpu2gpu_ms", 0)),
                                "t_gpu2gpu_ms": float(step_metadata.get("_t_gpu2gpu_ms", 0)),
                                "compute_ms": float(step_metadata.get("_compute_ms", 0)),
                                "t_gpu2cpu_ms": float(
                                    step_metadata.get("_t_gpu2cpu_ms", step_metadata.get("_serialize_ms", 0))
                                ),
                                "cpu_serialize_ms": float(step_metadata.get("_cpu_serialize_ms", 0)),
                                "step_total_ms": float(step_metadata.get("_step_total_ms", 0)),
                                "data_bytes": int(step_metadata.get("_data_bytes", 0)),
                                "gpu2gpu_bytes": float(step_metadata.get("_gpu2gpu_bytes", 0)),
                                "queue_wait_ms": queue_wait_ms,
                                "batch_size": int(step_metadata.get("_batch_size", 1)),
                                "token_increment": int(step_metadata.get("_token_increment", 1)),
                                "critical_path_exposed_ms": float(step_metadata.get("_critical_path_exposed_ms", 0)),
                                "sender_post_compute_exposed_ms": float(step_metadata.get("_sender_post_compute_exposed_ms", 0)),
                                "sender_gpu2cpu_exposed_ms": float(step_metadata.get("_sender_gpu2cpu_exposed_ms", 0)),
                                "sender_cpu2nic_exposed_ms": float(step_metadata.get("_sender_cpu2nic_exposed_ms", 0)),
                                "nic2nic_exposed_ms": float(step_metadata.get("_nic2nic_exposed_ms", 0)),
                                "receiver_dispatch_exposed_ms": float(step_metadata.get("_receiver_dispatch_exposed_ms", 0)),
                                "receiver_nic2cpu_exposed_ms": float(step_metadata.get("_receiver_nic2cpu_exposed_ms", 0)),
                                "receiver_cpu2gpu_exposed_ms": float(step_metadata.get("_receiver_cpu2gpu_exposed_ms", 0)),
                                "pipeline_overlap_breakdown_ready": int(step_metadata.get("_pipeline_overlap_breakdown_ready", 0)),
                                "upstream_sender_gpu2cpu_ms": float(step_metadata.get("_s2s_sender_gpu2cpu_ms", 0)),
                                "upstream_sender_cpu2nic_ms": float(step_metadata.get("_s2s_sender_cpu2nic_ms", 0)),
                                "upstream_wire_ms": float(step_metadata.get("_s2s_wire_ms", 0)),
                                "upstream_payload_bytes": int(step_metadata.get("_s2s_payload_bytes", 0)),
                            }
                            self._session_timing.setdefault(session_id, []).append(rec)
                        
                    end_iterate_rpc_inference_time=perf_counter() ###
                    # print('mean push time ', np.mean(push_time[4:])) ###
                    # print('finish iterate_rpc_inference time(sec) ', end_iterate_rpc_inference_time - end_cache_time) ###
                    # print_time_now('')
                    
                    # [MBPIPE] Cleanup async buffer if used
                    if output_buffer is not None:
                        try:
                            await output_buffer.stop(raise_on_error=True)
                        except Exception as e:
                            logger.warning(f"{MBPIPE_LOG_PREFIX} Buffer cleanup failed: {e}")
                            raise

                    # Ensure async push tasks complete before request finalization
                    await _drain_background_tasks()
            
            finally:
                # [MBPIPE_FIX] Clear offload state (CPU staging buffers) ONLY after the entire
                # request is complete. This ensures history is preserved during streaming.
                try:
                    if requested_backends:
                        cache_manager = requested_backends[0].cache_manager
                        if cache_manager is not None and hasattr(cache_manager, 'clear_offload_state'):
                            cache_manager.clear_offload_state()
                            logger.info(f"[MBPIPE_FIX] Cleared offload state after request completion")
                except Exception as e:
                    logger.warning(f"[MBPIPE_FIX] Failed to clear offload state: {e}")

                if background_tasks:
                    await asyncio.gather(*tuple(background_tasks), return_exceptions=True)
                await self._await_session_push_tasks(session_id)

                self._log_request("rpc_inference.close", requested_uids, context)
                if session_id:
                    self._emit_timing_summary(session_id, requested_uids)
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

    def _emit_timing_summary(self, session_id: str, requested_uids) -> None:
        records = self._session_timing.pop(session_id, [])
        comm_records = self._session_comm_timing.pop(session_id, {})
        if not records:
            return

        warmup = 1
        decode_records = records[warmup:] if len(records) > warmup else records
        if not decode_records:
            return

        blocks_desc = self._uids_to_block_span_label(requested_uids)
        compute_arr = np.array([r["compute_ms"] for r in decode_records], dtype=np.float64)
        gpu2cpu_arr = np.array([r["t_gpu2cpu_ms"] for r in decode_records], dtype=np.float64)
        step_arr = np.array([r["step_total_ms"] for r in decode_records], dtype=np.float64)
        queue_arr = np.array([r["queue_wait_ms"] for r in decode_records], dtype=np.float64)
        data_arr = np.array([r["data_bytes"] for r in decode_records], dtype=np.float64)
        nic2cpu_arr = np.array([r["t_nic2cpu_ms"] for r in decode_records], dtype=np.float64)
        cpu2gpu_arr = np.array([r["t_cpu2gpu_ms"] for r in decode_records], dtype=np.float64)
        gpu2gpu_arr = np.array([r.get("t_gpu2gpu_ms", 0.0) for r in decode_records], dtype=np.float64)
        gpu2gpu_bytes_arr = np.array([r.get("gpu2gpu_bytes", 0.0) for r in decode_records], dtype=np.float64)
        cpu_serialize_arr = np.array([r.get("cpu_serialize_ms", 0.0) for r in decode_records], dtype=np.float64)
        batch_arr = np.array([r.get("batch_size", 1) for r in decode_records], dtype=np.float64)
        token_arr = np.array([r.get("token_increment", 1) for r in decode_records], dtype=np.float64)

        total_compute = float(compute_arr.sum())
        total_step = float(step_arr.sum())
        total_nic2cpu = float(nic2cpu_arr.sum())
        total_cpu2gpu = float(cpu2gpu_arr.sum())
        total_gpu2cpu = float(gpu2cpu_arr.sum())
        total_host_io = total_nic2cpu + total_cpu2gpu + total_gpu2cpu
        compute_ratio = (total_compute / total_step * 100.0) if total_step > 0 else 0.0
        host_io_ratio = (total_host_io / total_step * 100.0) if total_step > 0 else 0.0
        gpu2cpu_ratio = (total_gpu2cpu / total_step * 100.0) if total_step > 0 else 0.0
        avg_bw = (data_arr.mean() / (gpu2cpu_arr.mean() / 1000.0) / 1e9) if gpu2cpu_arr.mean() > 0 else 0.0
        total_tokens = float(np.sum(batch_arr * token_arr))
        throughput_tok_s = (total_tokens / (total_step / 1000.0)) if total_step > 0 else 0.0
        inference_latency_ms = float(step_arr.mean()) if len(step_arr) > 0 else 0.0

        comm_summary = "\n  s2s_comm : no downstream push samples"
        cpu2nic_mean = 0.0
        nic2nic_mean = 0.0
        push_e2e_mean = 0.0
        avg_nic_bw = 0.0
        avg_nic_bw_gbps = 0.0
        comm_volume_kb = data_arr.mean() / 1024.0 if len(data_arr) > 0 else 0.0
        comm_volume_bytes = data_arr.mean() if len(data_arr) > 0 else 0.0
        total_cpu2nic = 0.0
        total_nic2nic = 0.0
        wire_arr = np.array([], dtype=np.float64)
        matched_comm_records = [comm_records[r["step_id"]] for r in decode_records if r.get("step_id") in comm_records]
        if matched_comm_records:
            cpu2nic_arr = np.array([r["t_cpu2nic_ms"] for r in matched_comm_records], dtype=np.float64)
            nic2nic_arr = np.array([r["t_nic2nic_ms"] for r in matched_comm_records], dtype=np.float64)
            push_e2e_arr = np.array([r["push_e2e_ms"] for r in matched_comm_records], dtype=np.float64)
            receiver_proc_arr = np.array([r["receiver_processing_ms"] for r in matched_comm_records], dtype=np.float64)
            wire_arr = np.array([r["wire_bytes"] for r in matched_comm_records], dtype=np.float64)

            total_cpu2nic = float(cpu2nic_arr.sum())
            total_nic2nic = float(nic2nic_arr.sum())
            total_comm = total_gpu2cpu + total_cpu2nic + total_nic2nic
            gpu2cpu_comm_ratio = (total_gpu2cpu / total_comm * 100.0) if total_comm > 0 else 0.0
            cpu2nic_ratio = (total_cpu2nic / total_comm * 100.0) if total_comm > 0 else 0.0
            nic2nic_ratio = (total_nic2nic / total_comm * 100.0) if total_comm > 0 else 0.0
            cpu2nic_mean = float(cpu2nic_arr.mean())
            nic2nic_mean = float(nic2nic_arr.mean())
            push_e2e_mean = float(push_e2e_arr.mean())
            avg_nic_bw = (wire_arr.mean() / (nic2nic_arr.mean() / 1000.0) / 1e6) if nic2nic_arr.mean() > 0 else 0.0
            avg_nic_bw_gbps = (wire_arr.mean() * 8.0 / (nic2nic_arr.mean() / 1000.0) / 1e9) if nic2nic_arr.mean() > 0 else 0.0
            comm_volume_kb = wire_arr.mean() / 1024.0 if len(wire_arr) > 0 else comm_volume_kb
            comm_volume_bytes = wire_arr.mean() if len(wire_arr) > 0 else comm_volume_bytes

            comm_summary = (
                f"\n  cpu2nic : mean={cpu2nic_arr.mean():.2f}ms  median={np.median(cpu2nic_arr):.2f}ms  "
                f"p95={np.percentile(cpu2nic_arr,95):.2f}ms  max={cpu2nic_arr.max():.2f}ms"
                f"\n  nic2nic : mean={nic2nic_arr.mean():.2f}ms  median={np.median(nic2nic_arr):.2f}ms  "
                f"p95={np.percentile(nic2nic_arr,95):.2f}ms  max={nic2nic_arr.max():.2f}ms"
                f"\n  push_e2e: mean={push_e2e_arr.mean():.2f}ms  median={np.median(push_e2e_arr):.2f}ms  "
                f"p95={np.percentile(push_e2e_arr,95):.2f}ms  max={push_e2e_arr.max():.2f}ms"
                f"\n  recv_proc: mean={receiver_proc_arr.mean():.2f}ms  median={np.median(receiver_proc_arr):.2f}ms  "
                f"p95={np.percentile(receiver_proc_arr,95):.2f}ms  max={receiver_proc_arr.max():.2f}ms"
                f"\n  s2s_ratio: gpu2cpu={gpu2cpu_comm_ratio:.1f}%  cpu2nic={cpu2nic_ratio:.1f}%  "
                f"nic2nic={nic2nic_ratio:.1f}%  avg_bw(nic)={avg_nic_bw:.1f}MB/s  wire_per_push={wire_arr.mean()/1024.0:.1f}KB"
            )

        pipeline_gpu2gpu_samples = []
        pipeline_gpu2gpu_bytes = []
        for rec in decode_records:
            sender_gpu2cpu_ms = float(rec.get("upstream_sender_gpu2cpu_ms", 0.0))
            sender_cpu2nic_ms = float(rec.get("upstream_sender_cpu2nic_ms", 0.0))
            upstream_wire_ms = float(rec.get("upstream_wire_ms", 0.0))
            if sender_gpu2cpu_ms <= 0.0 and sender_cpu2nic_ms <= 0.0 and upstream_wire_ms <= 0.0:
                continue
            pipeline_gpu2gpu_samples.append(
                sender_gpu2cpu_ms
                + sender_cpu2nic_ms
                + upstream_wire_ms
                + float(rec["t_nic2cpu_ms"])
                + float(rec["t_cpu2gpu_ms"])
            )
            pipeline_gpu2gpu_bytes.append(float(rec.get("upstream_payload_bytes", 0)))

        pipeline_gpu2gpu_arr = np.array(pipeline_gpu2gpu_samples, dtype=np.float64)
        pipeline_gpu2gpu_bytes_arr = np.array(pipeline_gpu2gpu_bytes, dtype=np.float64)
        pipeline_gpu2gpu_mean = float(pipeline_gpu2gpu_arr.mean()) if len(pipeline_gpu2gpu_arr) > 0 else 0.0
        pure_gpu2gpu_mean = float(gpu2gpu_arr.mean()) if len(gpu2gpu_arr) > 0 else 0.0
        pure_gpu2gpu_bw_mbps = (
            gpu2gpu_bytes_arr.mean() * 8.0 / (pure_gpu2gpu_mean / 1000.0) / 1e6
            if pure_gpu2gpu_mean > 0 and len(gpu2gpu_bytes_arr) > 0 and gpu2gpu_bytes_arr.mean() > 0
            else 0.0
        )
        local_gpu_staging_mean = float((gpu2cpu_arr + cpu2gpu_arr).mean()) if len(gpu2cpu_arr) > 0 else 0.0
        pure_gpu_compute_mean = float(compute_arr.mean()) if len(compute_arr) > 0 else 0.0
        pipeline_bw_mbps = (
            pipeline_gpu2gpu_bytes_arr.mean() * 8.0 / (pipeline_gpu2gpu_mean / 1000.0) / 1e6
            if pipeline_gpu2gpu_mean > 0 and len(pipeline_gpu2gpu_bytes_arr) > 0 and pipeline_gpu2gpu_bytes_arr.mean() > 0
            else 0.0
        )
        comm_volume_kb_runtime = float(comm_volume_bytes) / 1024.0 if comm_volume_bytes > 0 else 0.0
        upstream_sender_gpu2cpu_arr = np.array(
            [r.get("upstream_sender_gpu2cpu_ms", 0.0) for r in decode_records if r.get("upstream_sender_gpu2cpu_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        upstream_sender_cpu2nic_arr = np.array(
            [r.get("upstream_sender_cpu2nic_ms", 0.0) for r in decode_records if r.get("upstream_sender_cpu2nic_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        upstream_wire_arr = np.array(
            [r.get("upstream_wire_ms", 0.0) for r in decode_records if r.get("upstream_wire_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        upstream_payload_bytes_arr = np.array(
            [r.get("upstream_payload_bytes", 0.0) for r in decode_records if r.get("upstream_payload_bytes", 0.0) > 0.0],
            dtype=np.float64,
        )
        paper_gpu2cpu_mean = (
            float(upstream_sender_gpu2cpu_arr.mean()) if len(upstream_sender_gpu2cpu_arr) > 0 else float(gpu2cpu_arr.mean())
        )
        paper_cpu2nic_mean = (
            float(upstream_sender_cpu2nic_arr.mean()) if len(upstream_sender_cpu2nic_arr) > 0 else cpu2nic_mean
        )
        paper_nic2nic_mean = float(upstream_wire_arr.mean()) if len(upstream_wire_arr) > 0 else nic2nic_mean
        paper_comm_volume_bytes = (
            float(upstream_payload_bytes_arr.mean()) if len(upstream_payload_bytes_arr) > 0 else float(comm_volume_bytes)
        )
        paper_comm_volume_kb = paper_comm_volume_bytes / 1024.0 if paper_comm_volume_bytes > 0 else 0.0
        paper_net_latency_ms = push_e2e_mean if push_e2e_mean > 0 else paper_nic2nic_mean
        paper_net_bw_mbps = (
            paper_comm_volume_bytes * 8.0 / (paper_nic2nic_mean / 1000.0) / 1e6
            if paper_nic2nic_mean > 0 and paper_comm_volume_bytes > 0
            else 0.0
        )
        exposed_ready_count = sum(int(r.get("pipeline_overlap_breakdown_ready", 0)) for r in decode_records)
        critical_path_exposed_arr = np.array(
            [r.get("critical_path_exposed_ms", 0.0) for r in decode_records if r.get("critical_path_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        sender_gpu2cpu_exposed_arr = np.array(
            [r.get("sender_gpu2cpu_exposed_ms", 0.0) for r in decode_records if r.get("sender_gpu2cpu_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        sender_cpu2nic_exposed_arr = np.array(
            [r.get("sender_cpu2nic_exposed_ms", 0.0) for r in decode_records if r.get("sender_cpu2nic_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        nic2nic_exposed_arr = np.array(
            [r.get("nic2nic_exposed_ms", 0.0) for r in decode_records if r.get("nic2nic_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        receiver_nic2cpu_exposed_arr = np.array(
            [r.get("receiver_nic2cpu_exposed_ms", 0.0) for r in decode_records if r.get("receiver_nic2cpu_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        receiver_cpu2gpu_exposed_arr = np.array(
            [r.get("receiver_cpu2gpu_exposed_ms", 0.0) for r in decode_records if r.get("receiver_cpu2gpu_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        sender_post_compute_exposed_arr = np.array(
            [r.get("sender_post_compute_exposed_ms", 0.0) for r in decode_records if r.get("sender_post_compute_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        receiver_dispatch_exposed_arr = np.array(
            [r.get("receiver_dispatch_exposed_ms", 0.0) for r in decode_records if r.get("receiver_dispatch_exposed_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        critical_path_exposed_mean = (
            float(critical_path_exposed_arr.mean()) if len(critical_path_exposed_arr) > 0 else float(inference_latency_ms)
        )
        sender_gpu2cpu_exposed_mean = (
            float(sender_gpu2cpu_exposed_arr.mean()) if len(sender_gpu2cpu_exposed_arr) > 0 else paper_gpu2cpu_mean
        )
        sender_cpu2nic_exposed_mean = (
            float(sender_cpu2nic_exposed_arr.mean()) if len(sender_cpu2nic_exposed_arr) > 0 else paper_cpu2nic_mean
        )
        nic2nic_exposed_mean = (
            float(nic2nic_exposed_arr.mean()) if len(nic2nic_exposed_arr) > 0 else paper_nic2nic_mean
        )
        receiver_nic2cpu_exposed_mean = (
            float(receiver_nic2cpu_exposed_arr.mean()) if len(receiver_nic2cpu_exposed_arr) > 0 else float(nic2cpu_arr.mean())
        )
        receiver_cpu2gpu_exposed_mean = (
            float(receiver_cpu2gpu_exposed_arr.mean()) if len(receiver_cpu2gpu_exposed_arr) > 0 else float(cpu2gpu_arr.mean())
        )
        sender_post_compute_exposed_mean = (
            float(sender_post_compute_exposed_arr.mean()) if len(sender_post_compute_exposed_arr) > 0 else 0.0
        )
        receiver_dispatch_exposed_mean = (
            float(receiver_dispatch_exposed_arr.mean()) if len(receiver_dispatch_exposed_arr) > 0 else 0.0
        )

        n = len(decode_records)
        summary_message = (
            f"[TIMING_SUMMARY] blocks={blocks_desc} steps={n} (excl {warmup} warmup)\n"
            f"  nic2cpu : mean={nic2cpu_arr.mean():.2f}ms  median={np.median(nic2cpu_arr):.2f}ms  "
            f"p95={np.percentile(nic2cpu_arr,95):.2f}ms  max={nic2cpu_arr.max():.2f}ms\n"
            f"  cpu2gpu : mean={cpu2gpu_arr.mean():.2f}ms  median={np.median(cpu2gpu_arr):.2f}ms  "
            f"p95={np.percentile(cpu2gpu_arr,95):.2f}ms  max={cpu2gpu_arr.max():.2f}ms\n"
            f"  compute : mean={compute_arr.mean():.1f}ms  median={np.median(compute_arr):.1f}ms  "
            f"p95={np.percentile(compute_arr,95):.1f}ms  min={compute_arr.min():.1f}ms  max={compute_arr.max():.1f}ms\n"
            f"  gpu2cpu : mean={gpu2cpu_arr.mean():.2f}ms  median={np.median(gpu2cpu_arr):.2f}ms  "
            f"p95={np.percentile(gpu2cpu_arr,95):.2f}ms  max={gpu2cpu_arr.max():.2f}ms\n"
            f"  step_total: mean={step_arr.mean():.1f}ms  median={np.median(step_arr):.1f}ms  "
            f"p95={np.percentile(step_arr,95):.1f}ms  min={step_arr.min():.1f}ms  max={step_arr.max():.1f}ms\n"
            f"  queue_wait: mean={queue_arr.mean():.1f}ms  median={np.median(queue_arr):.1f}ms  "
            f"p95={np.percentile(queue_arr,95):.1f}ms  max={queue_arr.max():.1f}ms\n"
            f"  summary: inference_latency={inference_latency_ms:.2f}ms  throughput={throughput_tok_s:.2f}tok/s  "
            f"comm_volume={comm_volume_kb:.1f}KB  net_latency={push_e2e_mean:.2f}ms  net_bw={avg_nic_bw:.1f}MB/s\n"
            f"  ratio: compute={compute_ratio:.1f}%  host_io={host_io_ratio:.1f}%  gpu2cpu={gpu2cpu_ratio:.1f}%  "
            f"avg_bw(gpu2cpu)={avg_bw:.2f}GB/s  data_per_step={data_arr.mean()/1024:.1f}KB"
            f"{comm_summary}"
        )
        logger.info(summary_message)

        timing_table_line = (
            f"[TIMING_TABLE] blocks={blocks_desc} steps={n} "
            f"T_GPU->CPU={gpu2cpu_arr.mean():.2f}ms "
            f"T_CPU->NIC={cpu2nic_mean:.2f}ms "
            f"T_NIC->NIC={nic2nic_mean:.2f}ms "
            f"T_NIC->CPU={nic2cpu_arr.mean():.2f}ms "
            f"T_CPU->GPU={cpu2gpu_arr.mean():.2f}ms "
            f"InferenceLatency={inference_latency_ms:.2f}ms "
            f"Throughput={throughput_tok_s:.2f}tok/s "
            f"CommunicateVolume={comm_volume_kb:.1f}KB "
            f"T_GPU_Compute={compute_arr.mean():.2f}ms "
            f"NetLatency={push_e2e_mean:.2f}ms "
            f"NetBandwidth={avg_nic_bw:.2f}MB/s"
        )
        logger.info(timing_table_line)
        self._emit_unconditional_summary(timing_table_line)

        paper_timing_table_line = (
            f"[PAPER_TIMING_TABLE] blocks={blocks_desc} steps={n} "
            f"T_GPU->CPU={paper_gpu2cpu_mean:.2f}ms "
            f"T_CPU->NIC={paper_cpu2nic_mean:.2f}ms "
            f"T_NIC->NIC={paper_nic2nic_mean:.2f}ms "
            f"T_NIC->CPU={nic2cpu_arr.mean():.2f}ms "
            f"T_CPU->GPU={cpu2gpu_arr.mean():.2f}ms "
            f"InferenceLatency={inference_latency_ms:.2f}ms "
            f"Throughput={throughput_tok_s:.2f}tok/s "
            f"CommunicationVolume={paper_comm_volume_kb:.1f}KB "
            f"T_GPU_Compute={compute_arr.mean():.2f}ms "
            f"NetworkLatency={paper_net_latency_ms:.2f}ms "
            f"NetworkBandwidth={paper_net_bw_mbps:.2f}Mbps"
        )
        logger.info(paper_timing_table_line)
        self._emit_unconditional_summary(paper_timing_table_line)

        component_scope_line = (
            f"[PIPELINE_COMPONENT_VIEW] blocks={blocks_desc} steps={n} "
            f"sender_T_GPU->CPU_RAW={paper_gpu2cpu_mean:.2f}ms "
            f"sender_T_CPU->NIC_RAW={paper_cpu2nic_mean:.2f}ms "
            f"link_T_NIC->NIC_RAW={paper_nic2nic_mean:.2f}ms "
            f"receiver_T_NIC->CPU_RAW={nic2cpu_arr.mean():.2f}ms "
            f"receiver_T_CPU->GPU_RAW={cpu2gpu_arr.mean():.2f}ms "
            f"pipeline_overlap_affects_component_visibility=1"
        )
        logger.info(component_scope_line)
        self._emit_unconditional_summary(component_scope_line)

        exposed_component_line = (
            f"[PIPELINE_EXPOSED_VIEW] blocks={blocks_desc} steps={n} "
            f"EndToEndCriticalPathExposed={critical_path_exposed_mean:.2f}ms "
            f"sender_T_GPU->CPU_EXPOSED={sender_gpu2cpu_exposed_mean:.2f}ms "
            f"sender_T_CPU->NIC_EXPOSED={sender_cpu2nic_exposed_mean:.2f}ms "
            f"link_T_NIC->NIC_EXPOSED={nic2nic_exposed_mean:.2f}ms "
            f"receiver_T_NIC->CPU_EXPOSED={receiver_nic2cpu_exposed_mean:.2f}ms "
            f"receiver_T_CPU->GPU_EXPOSED={receiver_cpu2gpu_exposed_mean:.2f}ms "
            f"sender_post_compute_gap_EXPOSED={sender_post_compute_exposed_mean:.2f}ms "
            f"receiver_dispatch_EXPOSED={receiver_dispatch_exposed_mean:.2f}ms "
            f"overlap_breakdown_coverage={exposed_ready_count}/{n}"
        )
        logger.info(exposed_component_line)
        self._emit_unconditional_summary(exposed_component_line)

        pipeline_gpu_line = (
            f"[PIPELINE_GPU2GPU] blocks={blocks_desc} steps={n} "
            f"T_GPU->GPU_PIPE={pipeline_gpu2gpu_mean:.2f}ms "
            f"BW_GPU->GPU_PIPE={pipeline_bw_mbps:.2f}Mbps "
            f"T_GPU->GPU_PURE={pure_gpu2gpu_mean:.2f}ms "
            f"BW_GPU->GPU_PURE={pure_gpu2gpu_bw_mbps:.2f}Mbps "
            f"T_GPU_LOCAL_STAGING={local_gpu_staging_mean:.2f}ms "
            f"T_GPU_PURE_COMPUTE={pure_gpu_compute_mean:.2f}ms "
            f"T_CPU_SERIALIZE={cpu_serialize_arr.mean():.2f}ms "
            f"samples={len(pipeline_gpu2gpu_arr)}"
        )
        logger.info(pipeline_gpu_line)
        self._emit_unconditional_summary(pipeline_gpu_line)

        timing_note_line = (
            f"[TIMING_NOTE] blocks={blocks_desc} "
            f"T_GPU->GPU_PIPE=sender(T_GPU->CPU+T_CPU->NIC+wire)+receiver(T_NIC->CPU+T_CPU->GPU); "
            f"PAPER_TIMING_TABLE prefers upstream sender+wire fields on downstream stages when available; "
            f"T_GPU->GPU_PURE is same-host cuda->cuda transfer time collected inside task_pool .to(device); "
            f"T_GPU->CPU happens on sender after compute; T_CPU->NIC happens on sender before rpc_push; "
            f"T_NIC->NIC is one-hop wire time; T_NIC->CPU happens on receiver during deserialize/unpack; "
            f"T_CPU->GPU happens on receiver when runtime moves tensors to cuda; "
            f"component fields are raw local segment durations and may be overlapped by pipeline; "
            f"they are not additive to end-to-end latency; "
            f"PIPELINE_EXPOSED_VIEW reports the downstream critical-path-visible portion of each segment; "
            f"with full-batch PP (no micro-batching), EXPOSED is reported from observed per-step segments and coverage should be n/n; "
            f"with micro-batch PP, EXPOSED uses overlap attribution when available and otherwise falls back to RAW means; "
            f"T_NIC->CPU includes deserialize/unpack; "
            f"T_CPU->NIC includes request packing and pre-send prep; "
            f"InferenceLatency/Throughput are stage-local means; "
            f"CommunicationVolume is mean payload per decode step in KB; "
            f"NetworkLatency=push_e2e(send->ack), while T_NIC->NIC subtracts receiver processing; "
            f"NetworkBandwidth=payload_bits/T_NIC->NIC in Mbps."
        )
        logger.info(timing_note_line)
        self._emit_unconditional_summary(timing_note_line)

    def _extract_rpc_push_timing(
        self,
        response: Optional[runtime_pb2.ExpertResponse],
        *,
        sender_send_us: int,
        sender_ack_us: int,
        fallback_rtt_ms: float,
    ) -> Dict[str, float]:
        result = {
            "end_to_end_rtt_ms": max(0.0, float(sender_ack_us - sender_send_us) / 1000.0),
            "network_rtt_ms": max(0.0, float(fallback_rtt_ms)),
            "receiver_processing_ms": 0.0,
        }
        if response is None or not response.metadata:
            return result

        try:
            response_meta = MSGPackSerializer.loads(response.metadata)
        except Exception:
            return result
        if not isinstance(response_meta, dict):
            return result

        receiver_recv_us = self._to_int(response_meta.get("clock_sync_receiver_recv_us"), 0)
        receiver_ack_us = self._to_int(response_meta.get("clock_sync_receiver_ack_us"), 0)
        if receiver_recv_us <= 0 or receiver_ack_us < receiver_recv_us or sender_ack_us < sender_send_us:
            return result

        receiver_processing_ms = max(0.0, float(receiver_ack_us - receiver_recv_us) / 1000.0)
        end_to_end_rtt_ms = max(0.0, float(sender_ack_us - sender_send_us) / 1000.0)
        network_rtt_ms = max(0.0, end_to_end_rtt_ms - receiver_processing_ms)
        return {
            "end_to_end_rtt_ms": end_to_end_rtt_ms,
            "network_rtt_ms": network_rtt_ms,
            "receiver_processing_ms": receiver_processing_ms,
        }

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

    async def _iterate_inference_steps(
        self,
        first_request: runtime_pb2.ExpertRequest,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        session_id: Optional[str],
        requested_uids: Sequence[str],
        context: P2PContext,
    ) -> AsyncIterator[Tuple[runtime_pb2.ExpertRequest, dict]]:
        processed_step_ids = set()
        # [MBPIPE_FIX] Track step routing to avoid double-processing the same step through
        # both micro-batch queue path and direct request path.
        microbatch_step_ids = set()
        processed_microbatch_ids = set()
        n_pushes = n_late_pushes = 0
        request = first_request
        anext_task = get_push_task = None
        queue_wait_ms = 0.0
        queue_wait_start_us = 0
        queue_wait_end_us = 0
        queue_source = "initial"
        try:
            start_iterate_inference_steps_time = perf_counter()
            
            with self._managed_session(session_id) if session_id is not None else contextlib.nullcontext():
                while request is not None:
                    # Start fetching the NEXT request early so network/queue wait can overlap with
                    # current step processing in iterate_rpc_inference.
                    if anext_task is None:
                        anext_task = asyncio.create_task(anext(requests))
                    if get_push_task is None:
                        if session_id is not None:
                            get_push_task = asyncio.create_task(self._get_from_session_queue(session_id))
                        else:
                            get_push_task = asyncio.create_task(asyncio.Event().wait())  # Dummy never-ending task

                    # [MBPIPE] Check if this is a micro-batch queue item (dict with type="micro_batch")
                    if is_microbatch_queue_item(request):
                        # Yield micro-batch directly with type marker
                        mb_item = request
                        mb_metadata = mb_item.get("metadata", {}).copy()
                        mb_step_id = mb_metadata.get("step_id")
                        mb_idx = mb_item.get("mb_idx", 0)
                        skip_mb_item = False

                        # If this step was already processed through full-batch path, ignore late micro-batch pushes.
                        if mb_step_id is not None and mb_step_id in processed_step_ids and mb_step_id not in microbatch_step_ids:
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} iterate_steps: skipping late micro-batch "
                                f"(step_id={mb_step_id}, mb_idx={mb_idx}) because full-batch path already processed it"
                            )
                            request = None
                            skip_mb_item = True

                        # Idempotency at consume side: prevent duplicate enqueue/replay from being processed twice.
                        if not skip_mb_item:
                            mb_dedup_key = (mb_step_id, mb_idx)
                            if mb_dedup_key in processed_microbatch_ids:
                                logger.info(
                                    f"{MBPIPE_LOG_PREFIX} iterate_steps: skipping duplicate micro-batch "
                                    f"(step_id={mb_step_id}, mb_idx={mb_idx})"
                                )
                                request = None
                                skip_mb_item = True
                            else:
                                processed_microbatch_ids.add(mb_dedup_key)
                                if mb_step_id is not None:
                                    microbatch_step_ids.add(mb_step_id)

                        if not skip_mb_item:
                            mb_metadata["type"] = "micro_batch"
                            mb_metadata["mb_idx"] = mb_idx
                            mb_metadata["expected_num_mb"] = mb_item.get("expected_num_mb", 1)
                            mb_metadata["offset"] = mb_item.get("offset", 0)
                            mb_metadata["size"] = mb_item.get("size", 1)
                            mb_metadata["full_batch_size"] = mb_item.get("full_batch_size", 1)
                            mb_metadata["pushed"] = True
                            mb_metadata["_queue_wait_ms"] = float(queue_wait_ms)
                            mb_metadata["_queue_source"] = queue_source
                            mb_metadata["_queue_wait_start_us"] = int(queue_wait_start_us)
                            mb_metadata["_queue_wait_end_us"] = int(queue_wait_end_us)
                            
                            logger.debug(
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
                        skip_direct_request = False
                        
                        # [MBPIPE] Note: Micro-batch signal handling removed.
                        if metadata.get("is_mb_start_signal"):
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} iterate_steps: ignoring mb_start_signal (incompatible format)"
                            )
                            request = None
                            skip_direct_request = True
                        
                        if pushed and not skip_direct_request:
                            n_pushes += 1
                            self._log_request("rpc_inference.push", requested_uids, context, debug=f"session received push")

                        # [MBPIPE_FIX] If this step is already being handled via micro-batch queue,
                        # skip direct/full-batch request to avoid double compute and KV corruption.
                        if (not skip_direct_request) and step_id is not None and step_id in microbatch_step_ids:
                            logger.info(
                                f"{MBPIPE_LOG_PREFIX} iterate_steps: skipping direct request for step_id={step_id} "
                                f"because micro-batch path is active"
                            )
                            request = None
                            skip_direct_request = True

                        if (not skip_direct_request) and (step_id is None or step_id not in processed_step_ids):
                            metadata["_queue_wait_ms"] = float(queue_wait_ms)
                            metadata["_queue_source"] = queue_source
                            metadata["_queue_wait_start_us"] = int(queue_wait_start_us)
                            metadata["_queue_wait_end_us"] = int(queue_wait_end_us)
                            yield request, metadata
                            if step_id is not None:
                                processed_step_ids.add(step_id)
                        elif (not skip_direct_request) and pushed:
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
                    
                    # Wait for next request, coming either from stream or push queue.
                    wait_start_time = perf_counter()
                    queue_wait_start_us = self._now_us()
                    done, _ = await asyncio.wait(
                        [anext_task, get_push_task], timeout=self.step_timeout, return_when=asyncio.FIRST_COMPLETED
                    )
                    queue_wait_end_us = self._now_us()
                    queue_wait_ms = (perf_counter() - wait_start_time) * 1000.0
                    
                    # Prefer push_queue when both are ready to keep micro-batch pipeline flowing.
                    if get_push_task in done:
                        request = await get_push_task
                        get_push_task = None
                        queue_source = "push_queue"
                    elif anext_task in done:
                        request = await anext_task
                        anext_task = None
                        queue_source = "stream"
                    else:
                        self._log_request("rpc_inference.step", requested_uids, context, warning="timed out")
                        anext_task.cancel()
                        get_push_task.cancel()
                        return
        except Exception:
            raise
        finally:
            for pending_task in (anext_task, get_push_task):
                if pending_task is not None and not pending_task.done():
                    pending_task.cancel()


    async def rpc_push(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        """Directly push activation tensors from one server to another"""

        requested_uids = self._check_uids(request.uid)
        metadata = MSGPackSerializer.loads(request.metadata)
        receive_us = self._now_us()
        session_id = metadata["session_id"]
        
        # [MBPIPE] Check if this is a micro-batch push from cross-stage streaming
        is_microbatch_push = metadata.get("is_microbatch_push", False)
        
        if is_microbatch_push:
            # Handle micro-batch push: accumulate until we have all micro-batches
            await self._handle_microbatch_push(request, metadata, requested_uids, context)
            return self._build_rpc_push_ack_response(receive_us)
        
        # Original flow: put into session queue for normal processing
        if metadata.get("pushed"):
            sender_blocks = str(metadata.get("sender_blocks", "unknown"))
            receiver_blocks = str(metadata.get("receiver_blocks", "unknown"))
            sender_send_us = self._to_int(metadata.get("clock_sync_sender_send_us"), 0)
            sender_to_receiver_clock_offset_us = self._to_int(metadata.get("sender_to_receiver_clock_offset_us"), 0)
            sender_to_receiver_clock_samples = self._to_int(metadata.get("sender_to_receiver_clock_samples"), 0)
            clock_sync_ok = sender_to_receiver_clock_samples > 0
            raw_transfer_ms = (
                max(0.0, (receive_us - sender_send_us) / 1000.0)
                if sender_send_us > 0 and receive_us >= sender_send_us
                else -1.0
            )
            wire_ms = -1.0
            if clock_sync_ok and sender_send_us > 0:
                sender_send_local_us = sender_send_us + sender_to_receiver_clock_offset_us
                wire_ms = max(0.0, (receive_us - sender_send_local_us) / 1000.0)
            payload_bytes = sum(len(t.buffer) for t in request.tensors)
            metadata_bytes = len(request.metadata) if request.metadata else 0
            logger.info(
                f"[S2S_WIRE] step_id={metadata.get('step_id')} channel=full_batch "
                f"sender_blocks={sender_blocks} receiver_blocks={receiver_blocks} "
                f"payload_kb={payload_bytes / 1024.0:.2f} metadata_b={metadata_bytes} "
                f"raw_transfer_ms={raw_transfer_ms:.3f} wire_ms={wire_ms:.3f} "
                f"clock_sync={int(clock_sync_ok)}"
            )
            self._record_s2s_network_sample(
                channel="full_batch",
                sender_blocks=sender_blocks,
                receiver_blocks=receiver_blocks,
                payload_bytes=payload_bytes,
                metadata_bytes=metadata_bytes,
                raw_transfer_ms=raw_transfer_ms,
                wire_ms=wire_ms,
                clock_sync_ok=clock_sync_ok,
            )
            metadata["_s2s_sender_gpu2cpu_ms"] = float(
                metadata.get("s2s_sender_gpu2cpu_ms", metadata.get("_t_gpu2cpu_ms", metadata.get("_serialize_ms", 0.0)))
            )
            metadata["_s2s_sender_cpu2nic_ms"] = float(metadata.get("s2s_sender_cpu2nic_ms", 0.0))
            metadata["_s2s_wire_ms"] = float(wire_ms if wire_ms >= 0.0 else raw_transfer_ms if raw_transfer_ms >= 0.0 else 0.0)
            metadata["_s2s_payload_bytes"] = int(payload_bytes)
            request.metadata = MSGPackSerializer.dumps(metadata)
        self._log_request("rpc_push", requested_uids, context, debug=f"session_id={session_id}")
        self._put_into_session_queue(session_id, request)
        return self._build_rpc_push_ack_response(receive_us)
    
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
        start_from_position = metadata.get("start_from_position", None)
        receive_us = self._now_us()
        
        # Use total_micro_batches from metadata if available, otherwise calculate
        expected_num_mb = resolve_expected_num_microbatches(
            full_batch_size,
            total_micro_batches=metadata.get("total_micro_batches"),
        )
        
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
        metadata["s2s_receiver_receive_us"] = int(receive_us)

        # [S2S_WIRE] Sender->receiver micro-batch transport timing breakdown.
        # Uses sender->receiver clock offset when available to isolate pure wire time.
        sender_blocks = str(metadata.get("sender_blocks", "unknown"))
        receiver_blocks = str(metadata.get("receiver_blocks", "unknown"))
        sender_send_us = self._to_int(metadata.get("clock_sync_sender_send_us"), 0)
        sender_ser_start_us = self._to_int(metadata.get("s2s_sender_serialize_start_us"), 0)
        sender_ser_end_us = self._to_int(metadata.get("s2s_sender_serialize_end_us"), 0)
        sender_enqueue_us = self._to_int(metadata.get("s2s_sender_enqueue_us"), 0)
        push_timestamp_us = self._to_int(metadata.get("stage_push_timestamp_us"), 0)
        sender_compute_to_serialize_start_ms = 0.0
        try:
            sender_compute_to_serialize_start_ms = float(
                metadata.get("s2s_sender_compute_to_serialize_start_ms", 0.0)
            )
        except Exception:
            sender_compute_to_serialize_start_ms = 0.0
        sender_sem_wait_ms = 0.0
        try:
            sender_sem_wait_ms = float(metadata.get("s2s_sender_sem_wait_ms", 0.0))
        except Exception:
            sender_sem_wait_ms = 0.0

        sender_to_receiver_clock_offset_us = self._to_int(metadata.get("sender_to_receiver_clock_offset_us"), 0)
        sender_to_receiver_clock_rtt_us = max(0, self._to_int(metadata.get("sender_to_receiver_clock_rtt_us"), 0))
        sender_to_receiver_clock_samples = self._to_int(metadata.get("sender_to_receiver_clock_samples"), 0)
        clock_sync_ok = sender_to_receiver_clock_samples > 0

        sender_serialize_ms = (
            max(0.0, (sender_ser_end_us - sender_ser_start_us) / 1000.0)
            if sender_ser_start_us > 0 and sender_ser_end_us >= sender_ser_start_us
            else -1.0
        )
        sender_queue_ms = (
            max(0.0, (sender_send_us - sender_enqueue_us) / 1000.0)
            if sender_enqueue_us > 0 and sender_send_us >= sender_enqueue_us
            else -1.0
        )
        sender_prep_ms = (
            max(0.0, (sender_send_us - sender_ser_end_us) / 1000.0)
            if sender_ser_end_us > 0 and sender_send_us >= sender_ser_end_us
            else -1.0
        )
        sender_pre_send_wait_ms = sender_prep_ms
        sender_pre_send_post_enqueue_ms = sender_queue_ms
        sender_pre_send_misc_ms = (
            max(0.0, sender_pre_send_wait_ms - sender_sem_wait_ms - max(0.0, sender_pre_send_post_enqueue_ms))
            if sender_pre_send_wait_ms >= 0.0
            else -1.0
        )
        raw_transfer_ms = (
            max(0.0, (receive_us - push_timestamp_us) / 1000.0)
            if push_timestamp_us > 0 and receive_us >= push_timestamp_us
            else -1.0
        )

        wire_ms = -1.0
        e2e_from_serialize_end_ms = -1.0
        if clock_sync_ok and sender_send_us > 0:
            sender_send_local_us = sender_send_us + sender_to_receiver_clock_offset_us
            wire_ms = max(0.0, (receive_us - sender_send_local_us) / 1000.0)
            if sender_ser_end_us > 0:
                sender_ser_end_local_us = sender_ser_end_us + sender_to_receiver_clock_offset_us
                e2e_from_serialize_end_ms = max(0.0, (receive_us - sender_ser_end_local_us) / 1000.0)

        payload_bytes = sum(len(t.buffer) for t in request.tensors)
        metadata_bytes = len(request.metadata) if request.metadata else 0
        logger.info(
            f"[S2S_WIRE] step_id={step_id} mb_idx={int(mb_idx)} "
            f"sender_blocks={sender_blocks} receiver_blocks={receiver_blocks} "
            f"batch={int(mb_size)} payload_kb={payload_bytes/1024.0:.2f} metadata_b={metadata_bytes} "
            f"raw_transfer_ms={raw_transfer_ms:.3f} "
            f"sender_compute_to_serialize_start_ms={sender_compute_to_serialize_start_ms:.3f} "
            f"sender_serialize_ms={sender_serialize_ms:.3f} "
            f"sender_sem_wait_ms={sender_sem_wait_ms:.3f} "
            f"sender_queue_ms={sender_queue_ms:.3f} sender_prep_ms={sender_prep_ms:.3f} "
            f"sender_pre_send_wait_ms={sender_pre_send_wait_ms:.3f} "
            f"sender_pre_send_post_enqueue_ms={sender_pre_send_post_enqueue_ms:.3f} "
            f"sender_pre_send_misc_ms={sender_pre_send_misc_ms:.3f} "
            f"wire_ms={wire_ms:.3f} e2e_from_serialize_end_ms={e2e_from_serialize_end_ms:.3f} "
            f"clock_sync={int(clock_sync_ok)} "
            f"clock_offset_ms={sender_to_receiver_clock_offset_us/1000.0:.3f} "
            f"clock_rtt_ms={sender_to_receiver_clock_rtt_us/1000.0:.3f}"
        )
        self._record_s2s_network_sample(
            channel="micro_batch",
            sender_blocks=sender_blocks,
            receiver_blocks=receiver_blocks,
            payload_bytes=payload_bytes,
            metadata_bytes=metadata_bytes,
            raw_transfer_ms=raw_transfer_ms,
            wire_ms=wire_ms,
            clock_sync_ok=clock_sync_ok,
        )
        metadata["_s2s_sender_gpu2cpu_ms"] = float(
            metadata.get("s2s_sender_gpu2cpu_ms", sender_serialize_ms if sender_serialize_ms >= 0.0 else 0.0)
        )
        metadata["_s2s_sender_cpu2nic_ms"] = float(metadata.get("s2s_sender_cpu2nic_ms", sender_prep_ms if sender_prep_ms >= 0.0 else 0.0))
        metadata["_s2s_wire_ms"] = float(wire_ms if wire_ms >= 0.0 else raw_transfer_ms if raw_transfer_ms >= 0.0 else 0.0)
        metadata["_s2s_payload_bytes"] = int(payload_bytes)
        
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

        logger.debug(
            f"{MBPIPE_LOG_PREFIX} rpc_push: step_id={step_id}, mb_idx={mb_idx}, "
            f"start_from_position={start_from_position}, received={received_count}/{expected_num_mb}"
        )
        
        acc_key = f"{session_id}|{step_id}"
        file_received_count = 0
        wrote_file = False
        if (not self._enable_immediate_mb_queue) or self._store_mb_files_in_immediate:
            # Legacy fallback: keep file-based storage for reconstruction/diagnostics.
            tensor_bytes = [t.SerializeToString() for t in request.tensors]
            file_received_count = _store_microbatch_to_file(acc_key, mb_idx, tensor_bytes, metadata.copy())
            wrote_file = True
        
        if self._enable_immediate_mb_queue:
            # ========== IMMEDIATE QUEUING PATH (Step A) ==========
            # Put each micro-batch directly into session queue as a queue item
            metadata["s2s_receiver_queue_put_us"] = int(self._now_us())
            
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
            
            logger.debug(
                f"{MBPIPE_LOG_PREFIX} rpc_push: mb_idx={mb_idx} IMMEDIATELY queued to session "
                f"(received={received_count}/{expected_num_mb})"
            )
            
            # Cleanup tracking when all micro-batches for this step are queued.
            if received_count >= expected_num_mb:
                if wrote_file:
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
        self,
        request: runtime_pb2.ExpertRequest,
        serialized_outputs: Union[runtime_pb2.Tensor, Sequence[runtime_pb2.Tensor]],
        metadata: dict,
        raise_on_error: bool = False,
    ) -> None:
        # print('_push_outputs metadata ', metadata)
        push_start_time = perf_counter()
        next_peer_id = None
        next_session_id = None
        next_start = None
        next_end = None
        try:
            next_servers = metadata.get("next_servers")
            if not next_servers:
                logger.debug("[DEBUG] _push_outputs: No next_servers, returning early")
                return

            next_peer_id, next_session_id, next_start, next_end = next_servers[0]
            next_peer_id_str = str(next_peer_id)
            next_peer_id = PeerID.from_base58(next_peer_id)
            next_uid = CHAIN_DELIMITER.join(f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(next_start, next_end))
            sender_blocks = self._uids_to_block_span_label(request.uid)

            # Log cross-GPU transfer start
            if is_log_channel_enabled("cross_gpu_transfer_logs"):
                logger.info(
                    f"[CROSS_GPU_TRANSFER_START] FromBlocks={sender_blocks} ToBlocks={next_start}:{next_end} ToPeer={next_peer_id}"
                )

            # `serialized_outputs` carries the updated routing tensors for the
            # next stage. Regular decode emits a compact 3-tensor prefix
            # (hidden_states, keep_indices, need_pruning), while speculative
            # decoding emits a 6-tensor routing prefix that also includes
            # tree_attention_mask, kv_cache_position_ids and draft_tokens.
            # Reconstruct the downstream rpc_inference tensor layout according
            # to the original request metadata and keep control flags in
            # metadata when possible.
            normalized_outputs = self._normalize_serialized_tensors(serialized_outputs)
            if len(normalized_outputs) == 3:
                next_tensors = normalized_outputs + list(request.tensors[3:])
            elif len(normalized_outputs) == 6:
                inference_layout = metadata.get("inference_layout")
                if inference_layout == "spec_compact_v1":
                    need_pruning_next = deserialize_torch_tensor(normalized_outputs[2])
                    if torch.is_tensor(need_pruning_next) and need_pruning_next.numel() > 0:
                        next_need_pruning = int(bool(need_pruning_next.bool().any().item()))
                    else:
                        next_need_pruning = 0
                    next_tensors = [
                        normalized_outputs[0],
                        normalized_outputs[1],
                        normalized_outputs[3],
                        normalized_outputs[4],
                        normalized_outputs[5],
                        request.tensors[5],
                        request.tensors[6],
                        request.tensors[7],
                    ]
                else:
                    next_need_pruning = None
                    next_tensors = normalized_outputs + list(request.tensors[6:])
            else:
                raise ValueError(
                    f"Unexpected routing tensor count from upstream stage: {len(normalized_outputs)}"
                )
            next_metadata = metadata.copy()
            next_metadata.update(session_id=next_session_id, next_servers=next_servers[1:], pushed=True)
            if len(normalized_outputs) == 6 and metadata.get("inference_layout") == "spec_compact_v1":
                next_metadata["need_pruning"] = next_need_pruning
            next_metadata["sender_blocks"] = sender_blocks
            next_metadata["receiver_blocks"] = f"{next_start}:{next_end}"
            next_metadata["s2s_channel"] = "full_batch"
            next_metadata["s2s_sender_enqueue_us"] = int(self._now_us())
            clock_sync_estimate = self._get_clock_sync_estimate(next_peer_id_str)
            if clock_sync_estimate is not None:
                next_metadata["sender_to_receiver_clock_offset_us"] = clock_sync_estimate["offset_us"]
                next_metadata["sender_to_receiver_clock_rtt_us"] = clock_sync_estimate["rtt_us"]
                next_metadata["sender_to_receiver_clock_samples"] = clock_sync_estimate["samples"]
            sender_send_us = self._now_us()
            next_metadata["clock_sync_sender_send_us"] = sender_send_us
            t_gpu2cpu_ms = float(metadata.get("_t_gpu2cpu_ms", metadata.get("_serialize_ms", 0.0)))
            next_metadata["s2s_sender_gpu2cpu_ms"] = float(t_gpu2cpu_ms)

            stub = self.get_stub(self._p2p, next_peer_id)
            push_tensor_bytes = sum(len(t.buffer) for t in next_tensors)
            cpu2nic_prep_end = perf_counter()
            t_cpu2nic_ms = max(0.0, (cpu2nic_prep_end - push_start_time) * 1000.0)
            next_metadata["s2s_sender_cpu2nic_ms"] = float(t_cpu2nic_ms)
            serialized_next_metadata = MSGPackSerializer.dumps(next_metadata)
            push_metadata_bytes = len(serialized_next_metadata)
            rpc_request = runtime_pb2.ExpertRequest(uid=next_uid, tensors=next_tensors, metadata=serialized_next_metadata)

            nic2nic_start = perf_counter()
            response = await stub.rpc_push(rpc_request, timeout=self.request_timeout)
            nic2nic_end = perf_counter()
            sender_ack_us = self._now_us()
            rpc_timing = self._extract_rpc_push_timing(
                response,
                sender_send_us=sender_send_us,
                sender_ack_us=sender_ack_us,
                fallback_rtt_ms=(nic2nic_end - nic2nic_start) * 1000.0,
            )
            self._update_clock_sync_from_rpc_response(
                peer_id=next_peer_id_str,
                sender_send_us=sender_send_us,
                sender_ack_us=sender_ack_us,
                response=response,
            )
            transfer_time_ms = (nic2nic_end - push_start_time) * 1000.0
            transfer_bw_mbps = self._calc_mbps(push_tensor_bytes + push_metadata_bytes, transfer_time_ms)
            t_nic2nic_ms = float(rpc_timing["network_rtt_ms"])
            push_e2e_ms = float(rpc_timing["end_to_end_rtt_ms"])
            receiver_processing_ms = float(rpc_timing["receiver_processing_ms"])

            # T(GPU→CPU) comes from the compute step's serialization timing.
            compute_ms = float(metadata.get("_compute_ms", 0.0))
            data_bytes = int(metadata.get("_data_bytes", 0))

            total_comm_ms = t_gpu2cpu_ms + t_cpu2nic_ms + t_nic2nic_ms
            gpu2cpu_pct = (t_gpu2cpu_ms / total_comm_ms * 100) if total_comm_ms > 0 else 0.0
            cpu2nic_pct = (t_cpu2nic_ms / total_comm_ms * 100) if total_comm_ms > 0 else 0.0
            nic2nic_pct = (t_nic2nic_ms / total_comm_ms * 100) if total_comm_ms > 0 else 0.0

            critical_path_ms = compute_ms + total_comm_ms
            compute_critical_pct = (compute_ms / critical_path_ms * 100) if critical_path_ms > 0 else 0.0
            comm_critical_pct = (total_comm_ms / critical_path_ms * 100) if critical_path_ms > 0 else 0.0

            bw_nic_mbps = (push_tensor_bytes / (t_nic2nic_ms / 1000) / 1e6) if t_nic2nic_ms > 0 else 0.0
            bw_gpu2cpu_gbps = (data_bytes / (t_gpu2cpu_ms / 1000) / 1e9) if t_gpu2cpu_ms > 0 else 0.0

            step_id = metadata.get("step_id", "unknown")
            session_id = metadata.get("session_id")
            self._record_session_comm_timing(
                session_id,
                step_id,
                t_cpu2nic_ms=t_cpu2nic_ms,
                t_nic2nic_ms=t_nic2nic_ms,
                push_e2e_ms=push_e2e_ms,
                receiver_processing_ms=receiver_processing_ms,
                wire_bytes=push_tensor_bytes,
            )
            logger.info(
                f"[COMM_BREAKDOWN] step_id={step_id} "
                f"to_blocks={next_start}:{next_end} "
                f"T(GPU→CPU)={t_gpu2cpu_ms:.2f}ms({gpu2cpu_pct:.1f}%) "
                f"T(CPU→NIC)={t_cpu2nic_ms:.2f}ms({cpu2nic_pct:.1f}%) "
                f"T(NIC→NIC)={t_nic2nic_ms:.2f}ms({nic2nic_pct:.1f}%) "
                f"push_e2e={push_e2e_ms:.2f}ms "
                f"recv_proc={receiver_processing_ms:.2f}ms "
                f"total_comm={total_comm_ms:.2f}ms "
                f"compute={compute_ms:.2f}ms "
                f"critical_path: compute={compute_critical_pct:.1f}% comm={comm_critical_pct:.1f}% "
                f"BW(NIC)={bw_nic_mbps:.1f}MB/s BW(GPU→CPU)={bw_gpu2cpu_gbps:.1f}GB/s "
                f"wire_bytes={push_tensor_bytes}"
            )

            logger.info(f"[NETWORK_S2S] PUSH_COMPLETE | "
                       f"from_blocks={sender_blocks} | to_blocks={next_start}:{next_end} | "
                       f"tensor_size={push_tensor_bytes/1024:.2f}KB | "
                       f"metadata_size={push_metadata_bytes}B | "
                       f"transfer_time={transfer_time_ms:.2f}ms | "
                       f"approx_bw={transfer_bw_mbps:.2f}Mbps")
            
        except Exception:
            logger.warning(
                f"Failed to push outputs to peer_id={next_peer_id}, session_id={next_session_id}, blocks={next_start}:{next_end}:",
                exc_info=True,
            )
            if raise_on_error:
                raise

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
            is_spec_push = bool(metadata.get("is_spec_dec", False))

            # Speculative decoding requires strict full-batch context (tree/draft/kv alignment).
            # Do not use cross-stage micro-batch push for this mode.
            if is_spec_push:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} Cross-stage push skipped for speculative decoding "
                    f"(step_id={metadata.get('step_id')}, mb_idx={mb_idx})"
                )
                return
            
            next_peer_id, next_session_id, next_start, next_end = next_servers[0]
            next_peer_id_str = str(next_peer_id)
            
            # Log the push intent
            logger.debug(
                f"{MBPIPE_LOG_PREFIX} Cross-stage push: mb_idx={mb_idx}, "
                f"offset={mb_offset}, size={mb_size}, to={next_start}:{next_end}"
                f"{'' if enable_actual_push else ' (dry-run, set BLOOMBEE_ENABLE_CROSS_STAGE_PUSH=1 to enable)'}"
            )
            
            # Only actually send if the feature is enabled
            if not enable_actual_push:
                return
            
            next_peer_id = PeerID.from_base58(next_peer_id)
            next_uid = CHAIN_DELIMITER.join(f"{self.dht_prefix}{UID_DELIMITER}{i}" for i in range(next_start, next_end))
            
            # Serialize the micro-batch tensors
            outputs_schema = tuple(nested_flatten(requested_backends[-1].outputs_schema))
            sender_compute_end_us = self._to_int(metadata.get("stage_compute_end_timestamp_us"), 0)
            serialize_start_us = self._now_us()
            transport_phase = "prefill" if mb_hidden.ndim >= 2 and int(mb_hidden.shape[1]) > 1 else "decode"
            sender_blocks_str = str(metadata.get("sender_blocks", "unknown"))
            sender_blocks = sender_blocks_str
            push_blocks = f"{sender_blocks_str}->{next_start}:{next_end}"
            with transport_profile_scope() as push_transport_profile:
                serialized_hidden = serialize_torch_tensor(
                    mb_hidden.to(outputs_schema[0].dtype),
                    _s2s_output_compression if _s2s_output_compression is not None else outputs_schema[0].compression,
                    allow_inplace=True,
                    debug_context={
                        "phase": transport_phase,
                        "tensor_name": "hidden_states",
                        "source": "server",
                        "channel": "rpc_push_microbatch",
                        "blocks": push_blocks,
                        "batch": int(mb_size),
                    },
                )
                if mb_keep_indices is not None:
                    serialized_keep = serialize_torch_tensor(
                        mb_keep_indices.to(torch.int64),
                        outputs_schema[1].compression if len(outputs_schema) > 1 else runtime_pb2.CompressionType.NONE,
                        allow_inplace=True,
                        debug_context={
                            "phase": transport_phase,
                            "tensor_name": "keep_indices",
                            "source": "server",
                            "channel": "rpc_push_microbatch",
                            "blocks": push_blocks,
                            "batch": int(mb_size),
                        },
                    )
                else:
                    serialized_keep = serialize_torch_tensor(
                        torch.arange(mb_hidden.shape[1], dtype=torch.int64),
                        runtime_pb2.CompressionType.NONE,
                        allow_inplace=True,
                        debug_context={
                            "phase": transport_phase,
                            "tensor_name": "keep_indices",
                            "source": "server",
                            "channel": "rpc_push_microbatch",
                            "blocks": push_blocks,
                            "batch": int(mb_size),
                        },
                    )
            serialize_end_perf = perf_counter()
            serialize_end_us = self._now_us()
            sender_serialize_ms = max(0.0, (serialize_end_us - serialize_start_us) / 1000.0)
            t_gpu2cpu_ms = sender_serialize_ms
            sender_compute_to_serialize_start_ms = (
                max(0.0, (serialize_start_us - sender_compute_end_us) / 1000.0)
                if sender_compute_end_us > 0 and serialize_start_us >= sender_compute_end_us
                else -1.0
            )
            log_comp_ratio_event(
                logger,
                source="server",
                channel="rpc_push_microbatch",
                blocks=push_blocks,
                step_id=str(metadata.get("step_id", "unknown")),
                batch_size=int(mb_size),
                tensor_name="hidden_states",
                raw_bytes=tensor_raw_nbytes(mb_hidden),
                wire_bytes=len(serialized_hidden.buffer),
                nnz_ratio=tensor_nnz_ratio(mb_hidden),
                extra={
                    "mb_idx": int(mb_idx),
                    "phase": transport_phase,
                    "seq_tokens": int(mb_hidden.shape[1]) if mb_hidden.ndim >= 2 else 1,
                },
            )
            log_transport_profile_event(
                logger,
                source="server",
                channel="rpc_push_microbatch",
                blocks=push_blocks,
                step_id=str(metadata.get("step_id", "unknown")),
                batch_size=int(mb_size),
                stats=push_transport_profile,
                extra={
                    "mb_idx": int(mb_idx),
                    "phase": transport_phase,
                    "seq_tokens": int(mb_hidden.shape[1]) if mb_hidden.ndim >= 2 else 1,
                },
            )
            activation_raw_bytes = int(metadata.get("activation_raw_bytes", tensor_raw_nbytes(mb_hidden)))
            activation_wire_bytes = len(serialized_hidden.buffer)
            activation_ratio = (
                (activation_wire_bytes / activation_raw_bytes) if activation_raw_bytes > 0 else 1.0
            )
            kv_offload_bytes = int(metadata.get("kv_offload_bytes", 0))
            kv_prefetch_bytes = int(metadata.get("kv_prefetch_bytes", 0))
            kv_pcie_bytes = int(metadata.get("kv_pcie_bytes", 0))
            kv_to_activation_ratio = (
                (kv_pcie_bytes / activation_wire_bytes) if activation_wire_bytes > 0 else 0.0
            )
            logger.info(
                f"[ACTIVATION_XFER_CHECK] step_id={metadata.get('step_id', 'unknown')} "
                f"mb_idx={int(mb_idx)} blocks={sender_blocks}->{next_start}:{next_end} "
                f"batch={int(mb_size)} activation_raw_bytes={activation_raw_bytes} "
                f"activation_wire_bytes={activation_wire_bytes} activation_ratio={activation_ratio:.6f} "
                f"kv_offload_bytes={kv_offload_bytes} kv_prefetch_bytes={kv_prefetch_bytes} "
                f"kv_pcie_bytes={kv_pcie_bytes} kv_submit_ms={float(metadata.get('kv_pcie_submit_ms', 0.0)):.3f} "
                f"kv_block_ms={float(metadata.get('kv_pcie_block_ms', 0.0)):.3f} "
                f"kv_pcie_bw_mbps={float(metadata.get('kv_pcie_bw_mbps', 0.0)):.3f} "
                f"kv_gpu_alloc_mb={float(metadata.get('kv_gpu_alloc_mb', 0.0)):.3f} "
                f"kv_staging_peak_mb={float(metadata.get('kv_staging_peak_mb', 0.0)):.3f} "
                f"kv_to_activation_ratio={kv_to_activation_ratio:.6f} "
                f"invariant=1"
            )
            
            # Build metadata for micro-batch push
            push_metadata = {
                "session_id": next_session_id,
                "next_servers": next_servers[1:] if len(next_servers) > 1 else [],
                "pushed": True,
                # [MBPIPE] Micro-batch specific fields
                "is_microbatch_push": True,
                "micro_batch_idx": mb_idx,
                "micro_batch_offset": mb_offset,
                "micro_batch_size": mb_size,
                "full_batch_size": full_batch_size,
                "s2s_channel": "micro_batch",
                # Stable S1->S2 transport timing markers (sender clock domain)
                "sender_blocks": sender_blocks,
                "receiver_blocks": f"{next_start}:{next_end}",
                "s2s_sender_serialize_start_us": int(serialize_start_us),
                "s2s_sender_serialize_end_us": int(serialize_end_us),
                "s2s_sender_compute_to_serialize_start_ms": float(sender_compute_to_serialize_start_ms),
                "s2s_sender_gpu2cpu_ms": float(t_gpu2cpu_ms),
                "s2s_sender_cpu2nic_ms": float(t_cpu2nic_ms),
            }

            # [CLOCK_SYNC] Attach latest sender->receiver clock estimate for strict overlap correction
            # on downstream stage: downstream_local_time ~= upstream_time + offset_us.
            clock_sync_estimate = self._get_clock_sync_estimate(next_peer_id_str)
            if clock_sync_estimate is not None:
                push_metadata["sender_to_receiver_clock_offset_us"] = clock_sync_estimate["offset_us"]
                push_metadata["sender_to_receiver_clock_rtt_us"] = clock_sync_estimate["rtt_us"]
                push_metadata["sender_to_receiver_clock_samples"] = clock_sync_estimate["samples"]
            
            # Copy other relevant metadata
            # [CROSS_STAGE] Include timestamps for cross-stage overlap analysis
            for key in [
                "step_id",
                "max_length",
                "is_spec_dec",
                "need_pruning",
                "prefill_length",
                "stage_push_timestamp_us",
                "total_micro_batches",
                "stage_compute_start_timestamp_us",
                "stage_compute_end_timestamp_us",
                # [MBPIPE_FIX] Critical for KV correctness on downstream stage:
                # ensures each micro-batch of a step uses the same logical prefix.
                "start_from_position",
            ]:
                if key in metadata:
                    push_metadata[key] = metadata[key]
            
            stub = self.get_stub(self._p2p, next_peer_id)
            
            # Prioritize MB0 delivery to reduce per-step startup bubble on downstream stage.
            mb0_bypass_enabled = os.environ.get("BLOOMBEE_MB0_SEMAPHORE_BYPASS", "1") == "1"
            bypass_limiter = mb0_bypass_enabled and int(mb_idx) == 0
            acquired_slot = False
            sem_wait_time = 0.0
            if not bypass_limiter:
                sem_wait_time = await self._push_limiter.acquire()
                acquired_slot = True
                if sem_wait_time > 1.0:  # Only log if we had to wait
                    logger.info(
                        f"{MBPIPE_LOG_PREFIX} [FLOW_CONTROL] MB{mb_idx} waited {sem_wait_time:.1f}ms "
                        f"for push slot (limit={self._push_limiter.limit}, in_flight={self._push_limiter.in_flight})"
                    )
            else:
                logger.debug(
                    f"{MBPIPE_LOG_PREFIX} [FLOW_CONTROL] MB0 bypassed limiter "
                    f"(set BLOOMBEE_MB0_SEMAPHORE_BYPASS=0 to disable)"
                )

            # [ASYNC_PUSH] Fire-and-forget: don't await RPC response
            # This allows Stage 1 compute to continue immediately while data is sent in background.
            # These timestamps are used on the receiver to isolate pure wire latency.
            push_metadata["s2s_sender_sem_wait_ms"] = float(sem_wait_time)
            push_metadata["s2s_sender_enqueue_us"] = int(self._now_us())
            push_metadata["clock_sync_sender_send_us"] = int(push_metadata["s2s_sender_enqueue_us"])
            serialized_push_metadata = MSGPackSerializer.dumps(push_metadata)
            rpc_request = runtime_pb2.ExpertRequest(
                uid=next_uid,
                tensors=[serialized_hidden, serialized_keep],
                metadata=serialized_push_metadata,
            )
            t_cpu2nic_ms = max(0.0, (perf_counter() - serialize_end_perf) * 1000.0)
            push_tensor_bytes = len(serialized_hidden.buffer) + len(serialized_keep.buffer)
            
            # Create task for background sending - don't await
            send_task = asyncio.create_task(
                self._do_rpc_push_async(
                    stub,
                    rpc_request,
                    mb_idx,
                    push_start_time,
                    next_peer_id_str,
                    sender_session_id=metadata.get("session_id"),
                    step_id=push_metadata.get("step_id"),
                    to_blocks=f"{next_start}:{next_end}",
                    t_gpu2cpu_ms=t_gpu2cpu_ms,
                    t_cpu2nic_ms=t_cpu2nic_ms,
                    wire_bytes=push_tensor_bytes,
                    release_slot=acquired_slot,
                )
            )
            logger.info(
                f"[S2S_PUSH_BREAKDOWN] step_id={metadata.get('step_id', 'unknown')} "
                f"mb_idx={int(mb_idx)} sender_blocks={sender_blocks} receiver_blocks={next_start}:{next_end} "
                f"compute_to_serialize_start_ms={sender_compute_to_serialize_start_ms:.3f} "
                f"serialize_ms={sender_serialize_ms:.3f} "
                f"pre_send_wait_pending=1 "
                f"sem_wait_ms={float(sem_wait_time):.3f}"
            )
            
            # Track task to prevent garbage collection
            if not hasattr(self, '_background_push_tasks'):
                self._background_push_tasks = set()
            self._background_push_tasks.add(send_task)
            send_task.add_done_callback(self._background_push_tasks.discard)
            self._track_session_push_task(metadata.get("session_id"), send_task)
            
            queue_time = (perf_counter() - push_start_time) * 1000
            logger.debug(f"{MBPIPE_LOG_PREFIX} Micro-batch push queued in {queue_time:.1f}ms (sending in background)")
            
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
        peer_id: str,
        sender_session_id: Optional[str],
        step_id: Optional[str],
        to_blocks: str,
        t_gpu2cpu_ms: float,
        t_cpu2nic_ms: float,
        wire_bytes: int,
        *,
        release_slot: bool = True,
    ) -> None:
        """
        [ASYNC_PUSH] Actually perform the RPC push in background.
        
        This runs as a fire-and-forget task, allowing the main compute loop
        to continue without waiting for the network round-trip.
        """
        send_start = perf_counter()
        send_time = 0.0
        success = False
        try:
            payload_bytes = sum(len(t.buffer) for t in request.tensors)
            metadata_bytes = len(request.metadata) if request.metadata else 0
            sender_send_us = self._now_us()
            if request.metadata:
                try:
                    request_metadata = MSGPackSerializer.loads(request.metadata)
                    if isinstance(request_metadata, dict):
                        request_metadata["clock_sync_sender_send_us"] = sender_send_us
                        request.metadata = MSGPackSerializer.dumps(request_metadata)
                except Exception:
                    pass

            response = await stub.rpc_push(request, timeout=self.request_timeout)
            sender_ack_us = self._now_us()
            rpc_timing = self._extract_rpc_push_timing(
                response,
                sender_send_us=sender_send_us,
                sender_ack_us=sender_ack_us,
                fallback_rtt_ms=(perf_counter() - send_start) * 1000.0,
            )
            self._update_clock_sync_from_rpc_response(
                peer_id=peer_id,
                sender_send_us=sender_send_us,
                sender_ack_us=sender_ack_us,
                response=response,
            )
            total_time = (perf_counter() - queue_start_time) * 1000
            send_time = (perf_counter() - send_start) * 1000
            approx_bw_mbps = self._calc_mbps(payload_bytes + metadata_bytes, send_time)
            t_nic2nic_ms = float(rpc_timing["network_rtt_ms"])
            push_e2e_ms = float(rpc_timing["end_to_end_rtt_ms"])
            receiver_processing_ms = float(rpc_timing["receiver_processing_ms"])
            total_comm_ms = t_gpu2cpu_ms + t_cpu2nic_ms + t_nic2nic_ms
            bw_nic_mbps = (wire_bytes / (t_nic2nic_ms / 1000) / 1e6) if t_nic2nic_ms > 0 else 0.0
            self._record_session_comm_timing(
                sender_session_id,
                step_id,
                t_cpu2nic_ms=t_cpu2nic_ms,
                t_nic2nic_ms=t_nic2nic_ms,
                push_e2e_ms=push_e2e_ms,
                receiver_processing_ms=receiver_processing_ms,
                wire_bytes=wire_bytes,
            )
            logger.debug(
                f"{MBPIPE_LOG_PREFIX} [ASYNC_PUSH] MB{mb_idx} sent: "
                f"send={send_time:.1f}ms, total_from_queue={total_time:.1f}ms, "
                f"payload_kb={payload_bytes / 1024.0:.2f}, approx_bw={approx_bw_mbps:.2f}Mbps"
            )
            logger.info(
                f"[COMM_BREAKDOWN_MB] step_id={step_id or 'unknown'} mb_idx={mb_idx} "
                f"to_blocks={to_blocks} "
                f"T(GPU→CPU)={t_gpu2cpu_ms:.2f}ms "
                f"T(CPU→NIC)={t_cpu2nic_ms:.2f}ms "
                f"T(NIC→NIC)={t_nic2nic_ms:.2f}ms "
                f"push_e2e={push_e2e_ms:.2f}ms "
                f"recv_proc={receiver_processing_ms:.2f}ms "
                f"total_comm={total_comm_ms:.2f}ms "
                f"BW(NIC)={bw_nic_mbps:.1f}MB/s "
                f"wire_bytes={wire_bytes}"
            )
            success = True
        except Exception as e:
            logger.warning(
                f"{MBPIPE_LOG_PREFIX} [ASYNC_PUSH] MB{mb_idx} send failed: {e}"
            )
        finally:
            # Release slot and feed metrics to adaptive limiter.
            if release_slot and hasattr(self, "_push_limiter"):
                measured_send_ms = send_time if send_time > 0 else (perf_counter() - send_start) * 1000.0
                await self._push_limiter.release(send_time_ms=measured_send_ms, success=success)

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
        logical_full_batch_size: Optional[int] = None,
        max_length: int,
        timeout: Optional[float],
        force_full_batch_alloc: bool = False,
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

        # Micro-batching supports two modes:
        # - overlap-only (default): split execution but keep full logical KV cache
        # - GPU multiplexing (opt-in): shrink active GPU KV capacity and reuse slots
        from bloombee.utils.microbatch_config import get_micro_batch_size, get_micro_batch_config
        from bloombee.utils.memory_usage import log_mbpipe_memory, log_kv_cache_allocation, MemoryTracker
        
        mb_config = get_micro_batch_config()
        policy = cache_manager.offloading_policy
        max_supported_batch = policy.gpu_batch_size * max(1, int(getattr(policy, "num_gpu_batches", 1)))
        micro_batch_size = mb_config['micro_batch_size']
        working_slots = max(1, int(getattr(policy, "num_gpu_batches", 1)))
        logical_batch_size = (
            int(logical_full_batch_size)
            if logical_full_batch_size is not None and int(logical_full_batch_size) > 0
            else int(batch_size)
        )
        gpu_multiplexing_enabled = bool(mb_config.get('gpu_multiplexing', False))
        
        # [MBPIPE_DEBUG] Log the critical allocation decision
        logger.debug(f"[MBPIPE_ALLOC_DEBUG] ========================================")
        logger.debug(f"[MBPIPE_ALLOC_DEBUG] KV CACHE ALLOCATION DECISION POINT")
        logger.debug(f"[MBPIPE_ALLOC_DEBUG] ========================================")
        logger.debug(
            f"[MBPIPE_ALLOC_DEBUG] Input: request_batch={batch_size}, "
            f"logical_full_batch={logical_batch_size}, max_length={max_length}"
        )
        logger.debug(
            f"[MBPIPE_ALLOC_DEBUG] Config: mb_enabled={mb_config['enabled']}, "
            f"micro_batch_size={micro_batch_size}, mode={mb_config.get('mode', 'legacy')}"
        )
        logger.debug(f"[MBPIPE_ALLOC_DEBUG] Policy working capacity: {max_supported_batch}")
        
        if force_full_batch_alloc:
            # Speculative decoding currently requires full-batch KV residency for
            # correctness in verify path (tree mask/rotary/kv_valid alignment).
            # Do not multiplex KV cache for this session.
            alloc_batch_size = logical_batch_size
            logger.info(
                f"{MBPIPE_LOG_PREFIX} KV alloc mode: SPEC_FULL "
                f"(alloc_batch={alloc_batch_size}, request_batch={batch_size}, micro_batch={micro_batch_size})"
            )

        elif mb_config['enabled'] and micro_batch_size < logical_batch_size and gpu_multiplexing_enabled:
            # True GPU multiplexing:
            # - Keep logical full batch for scheduling
            # - Allocate a small number of GPU working slots
            # - Offload/prefetch swaps inactive per-micro-batch snapshots between CPU and GPU
            alloc_batch_size = min(logical_batch_size, micro_batch_size * working_slots)
            
            logger.debug(f"[MBPIPE_ALLOC_DEBUG] !!! MICRO-BATCHING ENABLED (GPU MULTIPLEXING) !!!")
            logger.debug(
                f"[MBPIPE_ALLOC_DEBUG] alloc_batch_size = {alloc_batch_size} "
                f"(working_slots={working_slots}, slot_batch_size={micro_batch_size})"
            )
            logger.debug(
                f"[MBPIPE_ALLOC_DEBUG] Full batch ({logical_batch_size}) will be processed in "
                f"{(logical_batch_size + micro_batch_size - 1) // micro_batch_size} micro-batches"
            )
            logger.debug(
                f"[MBPIPE_ALLOC_DEBUG] Micro-batches reuse {working_slots} GPU working slots; "
                f"inactive KV state is preserved via CPU snapshots"
            )
            
            # [MBPIPE_DEBUG] Calculate and log expected memory usage
            try:
                block_config = cache_manager.block_config
                log_kv_cache_allocation(
                    batch_size=logical_batch_size,
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
            # Legacy or overlap-only mode: keep a full logical KV cache allocation.
            alloc_batch_size = logical_batch_size
            if mb_config['enabled'] and micro_batch_size < logical_batch_size:
                logger.info(
                    f"{MBPIPE_LOG_PREFIX} KV alloc mode: OVERLAP_ONLY "
                    f"(alloc_batch={alloc_batch_size}, request_batch={batch_size}, "
                    f"micro_batch={micro_batch_size}, flexgen_offload=preserved)"
                )
                logger.debug(
                    "[MBPIPE_ALLOC_DEBUG] Micro-batching is enabled for overlap only; "
                    "keeping full-batch KV allocation to preserve FlexGen cache slicing"
                )
            else:
                logger.debug(f"[MBPIPE_ALLOC_DEBUG] Micro-batching disabled, alloc_batch_size={alloc_batch_size}")
            if logical_batch_size > max_supported_batch:
                raise AllocationFailed(
                    f"Requested batch size {logical_batch_size} exceeds server capacity "
                    f"{max_supported_batch}. Reduce client batch size or restart the "
                    f"server with a larger --batch_size value."
                )
        
        logger.debug(f"[MBPIPE_ALLOC_DEBUG] ========================================")
        
        # [MBPIPE_DEBUG] Call the memory savings diagnosis to explain current behavior
        try:
            from bloombee.utils.microbatch_config import log_memory_savings_diagnosis
            log_memory_savings_diagnosis(logger, logical_batch_size)
        except Exception as e:
            logger.debug(f"[MBPIPE_ALLOC_DEBUG] log_memory_savings_diagnosis failed: {e}")

        # Allocate cache descriptors for alloc_batch_size (= working slot capacity in micro-batch mode)
        descriptors = [backend.get_inference_cache_descriptors(alloc_batch_size, max_length) for backend in backends]

        logger.info(
            f"OFFLOAD: requesting KV allocation for {len(backends)} blocks, "
            f"alloc_batch={alloc_batch_size}, request_batch={batch_size}, "
            f"logical_batch={logical_batch_size}, max_length={max_length}"
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
            
            # In forked subprocesses, touching CUDA before it is initialized raises:
            # "Cannot re-initialize CUDA in forked subprocess". This cleanup is best-effort,
            # so only use CUDA APIs when the runtime is already initialized in this process.
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()
            
            logger.debug("Cleaned up temporary shared memory after warmup phase")
        except RuntimeError as e:
            if "cannot re-initialize cuda in forked subprocess" in str(e).lower():
                logger.debug("Skipping warmup shared memory CUDA cleanup in forked subprocess")
            else:
                logger.debug(f"Failed to cleanup warmup shared memory: {e}", exc_info=True)
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
