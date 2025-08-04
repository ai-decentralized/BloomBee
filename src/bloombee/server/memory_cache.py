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
from typing import AsyncContextManager, Dict, Optional, Sequence, Tuple, Any

import async_timeout
import torch
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from bloombee.data_structures import Handle
from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.misc import get_size_in_bytes

logger = get_logger(__name__)

# 创建专门的offloading调试logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


class DeviceInfo:
    """设备信息，用于offloading管理"""
    
    def __init__(self, device_type: str, device_id: Optional[str] = None, 
                 compression_config: Optional[Any] = None, offloaded: bool = False):
        self.device_type = device_type  # 'gpu', 'cpu', 'disk', 'mixed'
        self.device_id = device_id      # 'cuda:0', 'cpu', '/tmp/disk'
        self.compression_config = compression_config
        self.offloaded = offloaded
        self.sync_stream = None  # 用于异步同步的CUDA stream
    
    def __str__(self):
        return f"DeviceInfo(type={self.device_type}, id={self.device_id}, offloaded={self.offloaded})"


class UnifiedCache:
    """统一的缓存接口，包含past_key_value和设备信息"""
    
    def __init__(self, past_key_value: Optional[Tuple[torch.Tensor, ...]], 
                 device_info: DeviceInfo, cache_handles: Optional[Sequence[Handle]] = None):
        self.past_key_value = past_key_value  # 模型需要的缓存数据
        self.device_info = device_info        # offloading设备信息
        self.cache_handles = cache_handles    # MemoryCache的句柄
        self.metadata = {
            'layer_id': 0,
            'batch_id': 0,
            'position': 0,
            'created_time': time.time()
        }
    
    def __str__(self):
        return f"UnifiedCache(past_key_value={self.past_key_value is not None}, device_info={self.device_info})"


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, max_size_bytes: Optional[int], max_alloc_timeout: Optional[float] = None):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.max_alloc_timeout = max_alloc_timeout
        self._lock_metadata = mp.Lock()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=False)
        self._enqueued_size = mp.Value(ctypes.c_int64, 0, lock=True)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._allocated_tensors: Dict[Handle, torch.Tensor] = {}
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._lock_acquire_memory = mp.Lock()
        self._memory_freed_event = mp.Event()
        
        # 统一缓存管理
        self._unified_caches: Dict[Handle, UnifiedCache] = {}
        self._device_sync_streams: Dict[str, torch.cuda.Stream] = {}
        
        # 添加进程内句柄计数器
        self._local_handle_counter = 0

    @property
    def current_size_bytes(self) -> int:
        return self._current_size.value

    @current_size_bytes.setter
    def current_size_bytes(self, value: int):
        self._current_size.value = value

    @property
    def enqueued_size_bytes(self) -> int:
        return self._enqueued_size.value

    @enqueued_size_bytes.setter
    def enqueued_size_bytes(self, value: int):
        self._enqueued_size.value = value

    @property
    def bytes_left(self) -> int:
        return self.max_size_bytes - self.current_size_bytes

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    def store_unified_cache(self, unified_cache: UnifiedCache) -> Handle:
        """
        存储统一缓存接口
        
        Args:
            unified_cache: 统一缓存对象
            
        Returns:
            缓存句柄
        """
        if unified_cache.past_key_value is None:
            raise ValueError("past_key_value cannot be None")
        
        # 创建句柄
        handle = self._create_handle()
        
        # 存储到统一缓存字典
        self._unified_caches[handle] = unified_cache
        
        offload_logger.info(f" 存储UnifiedCache到句柄{handle}:")
        offload_logger.info(f"   - 当前_unified_caches大小: {len(self._unified_caches)}")
        offload_logger.info(f"   - 已存储的句柄: {list(self._unified_caches.keys())}")
        offload_logger.info(f"   - 句柄计数器: {self.handle_counter}")
        offload_logger.info(f"   - 是否覆盖现有句柄: {handle in self._unified_caches}")
        offload_logger.info(f"   - 进程ID: {os.getpid()}")
        offload_logger.info(f"   - 运行时PID: {self.runtime_pid}")
        
        # 计算缓存大小
        cache_size = 0
        if isinstance(unified_cache.past_key_value, (tuple, list)):
            for tensor in unified_cache.past_key_value:
                if isinstance(tensor, torch.Tensor):
                    cache_size += tensor.numel() * tensor.element_size()
                    tensor_handle = self._create_handle()
                    self._allocated_tensors[tensor_handle] = tensor
                    if unified_cache.cache_handles is None:
                        unified_cache.cache_handles = []
                    unified_cache.cache_handles.append(tensor_handle)
        
        # 更新当前大小
        self.current_size_bytes += cache_size
        
        offload_logger.info(f" MemoryCache存储UnifiedCache:")
        offload_logger.info(f"   - 句柄: {handle}")
        offload_logger.info(f"   - 设备: {unified_cache.device_info}")
        offload_logger.info(f"   - 缓存大小: {cache_size / 1024:.1f} KB")
        offload_logger.info(f"   - 总内存使用: {self.current_size_bytes / (1024*1024):.1f} MB")
        offload_logger.info(f"   - 剩余内存: {self.bytes_left / (1024*1024*1024):.1f} GB")
        
        logger.info(f"Stored unified cache with handle {handle}, device_info={unified_cache.device_info}")
        return handle

    def get_unified_cache(self, handle: Handle) -> Optional[UnifiedCache]:
        """
        获取统一缓存
        
        Args:
            handle: 缓存句柄
            
        Returns:
            统一缓存对象，如果不存在返回None
        """
        return self._unified_caches.get(handle)

    def sync_device_cache(self, unified_cache: UnifiedCache, target_device: str) -> UnifiedCache:
        """
        同步设备缓存，处理offloading
        
        Args:
            unified_cache: 统一缓存对象
            target_device: 目标设备
            
        Returns:
            同步后的统一缓存对象
        """
        if unified_cache.past_key_value is None:
            offload_logger.info(f"⚠️  past_key_value为None，跳过设备同步")
            return unified_cache
        
        offload_logger.info(f" MemoryCache设备同步:")
        offload_logger.info(f"   - 源设备: {unified_cache.device_info.device_id}")
        offload_logger.info(f"   - 目标设备: {target_device}")
        offload_logger.info(f"   - 当前offloaded: {unified_cache.device_info.offloaded}")
        offload_logger.info(f"   - past_key_value长度: {len(unified_cache.past_key_value)}")
        
        # 获取或创建同步流
        if target_device not in self._device_sync_streams:
            if target_device.startswith('cuda'):
                self._device_sync_streams[target_device] = torch.cuda.Stream()
                offload_logger.info(f"   - 创建CUDA流: {target_device}")
        
        # 同步设备
        with torch.cuda.stream(self._device_sync_streams.get(target_device)):
            if isinstance(unified_cache.past_key_value, (tuple, list)):
                synced_tensors = []
                for i, tensor in enumerate(unified_cache.past_key_value):
                    if isinstance(tensor, torch.Tensor):
                        offload_logger.info(f"   - 同步张量{i}: {tensor.device} -> {target_device}")
                        offload_logger.info(f"     - 张量形状: {tensor.shape}")
                        offload_logger.info(f"     - 张量dtype: {tensor.dtype}")
                        
                        synced_tensor = tensor.to(target_device, non_blocking=True)
                        synced_tensors.append(synced_tensor)
                        offload_logger.info(f"   - 张量{i}同步完成: {tensor.device} -> {synced_tensor.device}")
                    else:
                        synced_tensors.append(tensor)
                        offload_logger.info(f"   - 张量{i}: 非tensor对象，跳过同步")
                
                # 更新设备信息
                new_offloaded = target_device != 'cuda:0'
                new_device_info = DeviceInfo(
                    device_type=target_device.split(':')[0] if ':' in target_device else target_device,
                    device_id=target_device,
                    compression_config=unified_cache.device_info.compression_config,
                    offloaded=new_offloaded
                )
                
                offload_logger.info(f"   - 新offloaded状态: {new_offloaded}")
                offload_logger.info(f"   - 新设备信息: {new_device_info}")
                offload_logger.info(f" 设备同步完成")
                
                return UnifiedCache(
                    past_key_value=tuple(synced_tensors),
                    device_info=new_device_info,
                    cache_handles=unified_cache.cache_handles
                )
        
        offload_logger.warning(f"⚠️  past_key_value不是tuple/list，跳过同步")
        return unified_cache

    def _create_handle(self) -> Handle:
        """创建新的句柄"""
        with self._lock_metadata:
            # 使用进程内句柄计数器，避免多进程共享问题
            handle = self._local_handle_counter
            self._local_handle_counter += 1
            # 同时更新多进程共享的计数器
            self.handle_counter = self._local_handle_counter
            offload_logger.info(f" 创建新句柄: {handle} (本地计数器: {self._local_handle_counter})")
            offload_logger.info(f"   - 进程ID: {os.getpid()}")
            offload_logger.info(f"   - 运行时PID: {self.runtime_pid}")
            return handle

    @contextlib.asynccontextmanager
    async def allocate_cache(
        self, *descriptors: TensorDescriptor, timeout: float
    ) -> AsyncContextManager[Sequence[Handle]]:
        """
        Create a handle that is associated with buffers on unique device. If cache full, raises AllocationFailed.

        :param descriptors: one or more tensors tensor of this size, dtype, etc
        :param timeout: optional maximum time to wait for cache allocation; None (default) means no time limit

        :note: if descriptors reside on different devices, it is expected that they are approximately balanced across devices;
          if not, it will count maximum tensor allocation across devices for the purposes of size limit

        :note: This function should be called by connection handlers, it can be called concurrently from multiple processes.
        Furthermore, it can be called concurrently with at most one use_cache call in runtime.
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)
        max_alloc_size = self.get_allocation_size(*descriptors)

        gib = 1024**3
        cur_size, max_size = self.current_size_bytes, self.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / max_size * 100:.1f}%)"
        )

        alloc_task = asyncio.create_task(self._schedule_alloc(max_alloc_size, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self._free(max_alloc_size, alloc_task)

    @staticmethod
    def get_allocation_size(*descriptors: TensorDescriptor) -> int:
        """Return the memory size (bytes) to be allocated on a device. If there are many devices, return maximum"""
        alloc_size_by_device = {}
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] = alloc_size_by_device.get(descr.device, 0) + tensor_size
        return max(alloc_size_by_device.values())

    async def _schedule_alloc(
        self, alloc_size: int, *descriptors: TensorDescriptor, timeout: Optional[float]
    ) -> Sequence[Handle]:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """
        try:
            async with self._wait_for_free_memory(alloc_size, timeout):
                with self._lock_metadata:
                    handles = tuple(int(self.handle_counter) + i for i in range(len(descriptors)))
                    self.current_size_bytes += alloc_size
                    self.handle_counter += len(handles)  # note: this will eventually overflow and it is okay
                    self._pipe_send.send((handles, descriptors))
                    return handles
        except TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} (timeout={timeout})")

    @contextlib.asynccontextmanager
    async def _wait_for_free_memory(self, alloc_size: int, timeout: Optional[float]):
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        with self._enqueued_size.get_lock():
            self._enqueued_size.value += alloc_size
        allocated = False
        try:
            context_manager = async_timeout.timeout(timeout) if timeout != 0 else contextlib.AsyncExitStack()
            # contextlib.AsyncExitStack() is used as a null context here
            async with context_manager:
                if timeout == 0 and self.current_size_bytes + self.enqueued_size_bytes > self.max_size_bytes:
                    raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                async with enter_asynchronously(self._lock_acquire_memory):
                    if self.current_size_bytes + alloc_size > self.max_size_bytes:
                        if timeout == 0:
                            raise AllocationFailed(f"Could not allocate {alloc_size} bytes immediately: out of memory")
                        elapsed_time = time.perf_counter() - start_time
                        remaining_timeout = max(0.0, timeout - elapsed_time) if timeout is not None else None
                        await loop.run_in_executor(None, self._wait_until_available, alloc_size, remaining_timeout)

                allocated = True
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size
                yield
        except asyncio.TimeoutError:
            raise AllocationFailed(f"Could not allocate {alloc_size} within {timeout} seconds")
        finally:
            if not allocated:
                with self._enqueued_size.get_lock():
                    self._enqueued_size.value -= alloc_size

    def _free(self, alloc_size: int, alloc_task: asyncio.Task):
        if alloc_task.exception() is not None:
            return
        handles = alloc_task.result()

        with self._lock_metadata:
            self._pipe_send.send((handles, None))  # signal runtime to free these handles
            self.current_size_bytes -= alloc_size
        self._memory_freed_event.set()

    def _wait_until_available(self, allocated_size: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        timeout = timeout if timeout != float("inf") else None
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.current_size_bytes + allocated_size > self.max_size_bytes:
            remaining_time = None if timeout is None else deadline - time.perf_counter()
            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
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
        while self._pipe_recv.poll():
            recv_handles, recv_data = self._pipe_recv.recv()
            if recv_data is not None:  # create new tensors
                assert len(recv_handles) == len(recv_data)
                for handle, descr in zip(recv_handles, recv_data):
                    self._allocated_tensors[handle] = descr.make_zeros()
                    assert handle in self._allocated_tensors, f"Sanity check failed: no such handle ({handle})"
            else:  # delete tensors by handle
                for handle in recv_handles:
                    if handle not in self._allocated_tensors:
                        logger.warning(
                            f"Sanity check failed: asked to delete handle {handle}, but there is no such handle"
                        )
                    self._allocated_tensors.pop(handle, None)
        yield tuple(self._allocated_tensors[handle] for handle in handles)


class AllocationFailed(Exception):
    pass
