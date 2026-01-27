"""
A pytorch memory cache that can be allocated by ConnectionHandler (on cpu) and used over multiple calls to Runtime.

For now, the only purpose of this code is to ensure that allocated memory will be deleted properly.

"""
import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import time

import logging
from typing import AsyncContextManager, Dict, Optional, Sequence, Tuple, Union, Any, List

import async_timeout
import torch
import dataclasses
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

# Optimize shared memory usage: configure multiprocessing to use smaller buffers
# This helps reduce /dev/shm peak usage during warmup/prefill phase
if hasattr(mp, 'set_sharing_strategy'):
    # Use 'file_system' instead of 'file_descriptor' to reduce shared memory usage
    # Note: This may have slight performance impact but significantly reduces /dev/shm usage
    try:
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        # Already set, ignore
        pass


from bloombee.data_structures import Handle
from transformers import PretrainedConfig

from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.misc import get_size_in_bytes
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.task import Task

from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice, DeviceType, general_copy
from bloombee.flexgen_utils.compression import TorchCompressedDevice, general_copy_compressed
from bloombee.flexgen_utils.utils import torch_dtype_to_np_dtype
import numpy as np

logger = get_logger(__name__)

# Create dedicated offloading debug logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, 
                 max_size_tokens: Optional[int], 
                 max_alloc_timeout: Optional[float] = None, 
                 policy: Optional[Policy] = None, 
                 block_config: Optional[PretrainedConfig] = None,
                 device: Any = None):
        logger.info(f"max_size_tokens: {max_size_tokens}")
        self.max_size_tokens = max_size_tokens if max_size_tokens is not None else (2048)
        self.max_alloc_timeout = max_alloc_timeout
        self._lock_metadata = mp.Lock()
        self._current_tokens = mp.Value(ctypes.c_int64, 0, lock=False)
        self._enqueued_tokens = mp.Value(ctypes.c_int64, 0, lock=True)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._allocated_tensors: Dict[Handle, Any] = {}
        self._handle_size_bytes: Dict[Handle, int] = {}
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._lock_acquire_memory = mp.Lock()
        self._memory_freed_event = mp.Event()
        

        # flexgen' offloading depends on the task parameter, we need to mock a temp task variable for memory allocation without changing the data structure of flexgen.
        self.allocation_policy = policy
        self.block_config = block_config
        self.device = device
        

    @property
    def current_size_tokens(self) -> int:
        return self._current_tokens.value

    @current_size_tokens.setter
    def current_size_tokens(self, value: int):
        self._current_tokens.value = value

    @property
    def enqueued_size_tokens(self) -> int:
        return self._enqueued_tokens.value

    @enqueued_size_tokens.setter
    def enqueued_size_tokens(self, value: int):
        self._enqueued_tokens.value = value

    @property
    def tokens_left(self) -> int:
        return self.max_size_tokens - self.current_size_tokens

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    async def _schedule_alloc(
        self, alloc_tokens_num: int, *descriptors: TensorDescriptor, timeout: Optional[float]
    ) -> Sequence[Handle]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """
        try:
            async with self._wait_for_free_memory(alloc_tokens_num, timeout):
                with self._lock_metadata:
                    handles = tuple(int(self.handle_counter) + i for i in range(len(descriptors)))
                    self.current_size_tokens += alloc_tokens_num
                    self.handle_counter += len(handles)  # note: this will eventually overflow and it is okay
                    self._pipe_send.send((handles, descriptors))
                    return handles
        except TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_tokens_num} (timeout={timeout})")

    @contextlib.asynccontextmanager
    async def _wait_for_free_memory(self, alloc_tokens_num: int, timeout: Optional[float]):
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        with self._enqueued_tokens.get_lock():
            self._enqueued_tokens.value += alloc_tokens_num
        allocated = False
        try:
            context_manager = async_timeout.timeout(timeout) if timeout != 0 else contextlib.AsyncExitStack()
            # contextlib.AsyncExitStack() is used as a null context here
            async with context_manager:
                if timeout == 0 and self.current_size_tokens + self.enqueued_size_tokens > self.max_size_tokens:
                    raise AllocationFailed(f"Could not allocate {alloc_tokens_num} tokens cache immediately: out of memory")
                async with enter_asynchronously(self._lock_acquire_memory):
                    if self.current_size_tokens + alloc_tokens_num > self.max_size_tokens:
                        if timeout == 0:
                            raise AllocationFailed(f"Could not allocate {alloc_tokens_num} tokens cache immediately: out of memory")
                        elapsed_time = time.perf_counter() - start_time
                        remaining_timeout = max(0.0, timeout - elapsed_time) if timeout is not None else None
                        await loop.run_in_executor(None, self._wait_until_available, alloc_tokens_num, remaining_timeout)

                allocated = True
                with self._enqueued_tokens.get_lock():
                    self._enqueued_tokens.value -= alloc_tokens_num
                yield
        except asyncio.TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_tokens_num} within {timeout} seconds")
        finally:
            if not allocated:
                with self._enqueued_tokens.get_lock():
                    self._enqueued_tokens.value -= alloc_tokens_num

    def _free(self, alloc_tokens_num: int, alloc_task: asyncio.Task):
        if alloc_task.exception() is not None:
            return
        handles = alloc_task.result()

        with self._lock_metadata:
            self._pipe_send.send((handles, None))  # signal runtime to free these handles
            self.current_size_tokens -= alloc_tokens_num
        self._memory_freed_event.set()

    def _wait_until_available(self, alloc_tokens_num: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if alloc_tokens_num > self.max_size_tokens:
            raise AllocationFailed(
                f"Could not allocate {alloc_tokens_num} tokens cache, max cache size = {self.max_size_tokens} tokens"
            )
        timeout = timeout if timeout != float("inf") else None
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.current_size_tokens + alloc_tokens_num > self.max_size_tokens:
            remaining_time = None if timeout is None else deadline - time.perf_counter()
            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {alloc_tokens_num} tokens cache in {timeout} seconds"
                )
            self._memory_freed_event.clear()

    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        """
        Return one or more tensors previously allocated with allocate_cache,

        :note: This method is called by ModuleBackend in runtime: a single process with NO process parallelism.
        However, runtime may call use_cache concurrently with one or more connection handlers calling allocate_cache
        """
        assert os.getpid() == self.runtime_pid
        # note: this specific function is not concurrent, so you can safely allocate/offload/defragment data here

        # read creation/deletion requests from connection handlers
        def _process_message(recv_handles, recv_data):
            if recv_data is not None:  # create new tensors
                assert len(recv_handles) == len(recv_data)
                for handle, descr in zip(recv_handles, recv_data):
                    seq_len = descr.shape[-1] if descr.shape else 0
                    # [MBPIPE] Extract batch_size from descriptor (first dim)
                    # descriptor shape: (batch_size, num_heads, head_dim, max_length)
                    descr_batch_size = descr.shape[0] if descr.shape else 1
                    
                    self.mocked_task = Task(
                        inputs=None,
                        prompt_len=1,
                        gen_len=max(1, seq_len),
                        cut_gen_len=None,
                        do_sample=False,
                        temperature=0,
                        stop=None,
                        top_p=None,
                    )
                    # [MBPIPE] Override policy.gpu_batch_size with descriptor's batch_size
                    # This ensures KV cache is allocated for full batch, not micro-batch
                    from dataclasses import replace as dataclass_replace
                    override_policy = dataclass_replace(self.allocation_policy, gpu_batch_size=descr_batch_size)
                    logger.info(f"[MB_CACHE_ALLOC] Allocating cache: descr_batch_size={descr_batch_size}, "
                               f"seq_len={seq_len}, shape={descr.shape}")
                    
                    self._allocated_tensors[handle] = self.device.init_cache_one_gpu_batch(
                        self.block_config, self.mocked_task, override_policy
                    )
            else:  # delete tensors by handle
                for handle in recv_handles:
                    if handle not in self._allocated_tensors:
                        logger.warning(
                            f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                        )
                    self._allocated_tensors.pop(handle, None)

        while self._pipe_recv.poll():
            _process_message(*self._pipe_recv.recv())

        # Ensure all requested handles are materialized; wait synchronously if needed
        missing_handles = [handle for handle in handles if handle not in self._allocated_tensors]
        while missing_handles:
            _process_message(*self._pipe_recv.recv())
            missing_handles = [handle for handle in handles if handle not in self._allocated_tensors]

        yield tuple(self._allocated_tensors[handle] for handle in handles)
        

class AllocationFailed(Exception):
    pass

class KVCacheMetadata:
    device: Any               # On which device
    offloaded: bool = False             # Whether offloaded to CPU
    # TODO: add more device info


@dataclasses.dataclass(frozen=True)
class AdaptedKVCache:
    kvs: Sequence[torch.Tensor]
    device: KVCacheMetadata
