
from dataclasses import dataclass
import contextlib
import asyncio
import torch
import os
from typing import Optional, Tuple, AsyncContextManager, Sequence

from bloombee.server.memory_cache import MemoryCache, AdaptedKVCache, KVCacheMetadata
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import DeviceType

from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from bloombee.data_structures import Handle
from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.misc import get_size_in_bytes

from transformers import PretrainedConfig


logger = get_logger(__name__)


class KVCacheManager:
    def __init__(self, cache_max_size: int, 
                 max_alloc_timeout: int, 
                 policy: Policy, 
                 env: ExecutionEnv,
                 block_config: PretrainedConfig):
        # 初始化为二维数组结构
        self.env = env
        self.runtime_pid = os.getpid()
        self.cache = MemoryCache(cache_max_size, max_alloc_timeout, policy, block_config, self.get_cache_device(policy))
        self.offloading_policy = policy
        self.attention_compute = (self.env.cpu if policy.cpu_cache_compute
                                  else self.env.gpu)
        self.block_config = block_config
        self.max_alloc_timeout = max_alloc_timeout
        self._active_cache_tensors_stack = []
        
        
    def get_cache_device(self, policy):
        if policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        return device

    def clear(self):
        # for b in range(self.max_batch_size):
        #     for l in range(self.num_layers):
        #         self.cache[b][l] = None
        # No-op for now; handles are freed by context manager or force_free
        return
    
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
        if self.max_alloc_timeout is not None and timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)

        max_alloc_size = self.get_allocation_size(*descriptors)

        gib = 1024**3
        cur_size, max_size = self.current_size_bytes, self.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        used_pct = (cur_size / max_size * 100.0) if max_size != 0 and max_size != 2**64 - 1 else 0.0
        logger.info(
            f"rpc_inference.wait_for_alloc(size={max_alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({used_pct:.1f}%)"
        )

        alloc_task = asyncio.create_task(self.cache._schedule_alloc(max_alloc_size, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={max_alloc_size / gib:.2f} GiB)")
            yield handles
        finally:
            self.cache._free(max_alloc_size, alloc_task)
            
            
    @staticmethod
    def get_allocation_size(*descriptors: TensorDescriptor) -> int:
        """Return the memory size (bytes) to be allocated on a device. If there are many devices, return maximum"""
        alloc_size_by_device = {}
        for descr in descriptors:
            tensor_size = descr.numel() * get_size_in_bytes(descr.dtype)
            alloc_size_by_device[descr.device] = alloc_size_by_device.get(descr.device, 0) + tensor_size
        return max(alloc_size_by_device.values())
        
    
    def add_cache(self, kvs: AdaptedKVCache, start_position: int):
        self._write_kvs(kvs, start_position)
                
    def update_cache(
        self, new_kvs: AdaptedKVCache, start_position: int
    ):
        self._write_kvs(new_kvs, start_position)
    
    def bytes_left(self) -> int:
        return self.cache.bytes_left

    @property
    def current_size_bytes(self) -> int:
        return self.cache.current_size_bytes

    @property
    def max_size_bytes(self) -> int:
        return self.cache.max_size_bytes
    
    def select_cache(
        self,
        prefix_length: int,
        hypo_ids: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return aggregated past_key_value (K_all, V_all) for current active cache.

        - Aggregates over shards along head dimension
        - Optionally reorders batch by hypo_ids
        - Slices to prefix_length along sequence dimension

        Returns None if prefix_length <= 0
        """
        assert self._active_cache_tensors_stack, "select_cache called outside of use_cache"
        if prefix_length <= 0:
            return None

        cache_tensors = self._active_cache_tensors_stack[-1]
        keys_per_shard = []
        values_per_shard = []

        for key_shard, value_shard in zip(cache_tensors[0::2], cache_tensors[1::2]):
            # key_shard: [B, H, D, Lmax] -> slice & permute -> [B, H, Lp, D]
            # value_shard: [B, H, Lmax, D] -> slice -> [B, H, Lp, D]
            k = key_shard[:, :, :, :prefix_length].permute(0, 1, 3, 2)
            v = value_shard[:, :, :prefix_length, :]
            if hypo_ids is not None and isinstance(hypo_ids, torch.Tensor) and hypo_ids.numel() > 0:
                index = hypo_ids.to(k.device)
                k = k.index_select(0, index)
                v = v.index_select(0, index)
            keys_per_shard.append(k)
            values_per_shard.append(v)

        if len(keys_per_shard) == 1:
            return keys_per_shard[0], values_per_shard[0]

        key_all = torch.cat(keys_per_shard, dim=1)
        value_all = torch.cat(values_per_shard, dim=1)
        return key_all, value_all
    
    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        with self.cache.use_cache(*handles) as cache_tensors:
            # Keep underlying tensors in the stack for centralized writes,
            # but yield clones to callers to prevent accidental in-place edits
            self._active_cache_tensors_stack.append(cache_tensors)
            try:
                safe_views = tuple(t.detach().clone() for t in cache_tensors)
                yield safe_views
            finally:
                self._active_cache_tensors_stack.pop()

    def delete_cache(self, *handles: Handle):
        """Explicitly delete cache handles to free space early."""
        try:
            self.cache.force_free(*handles)
        except Exception as e:
            logger.warning(f"OFFLOAD: delete_cache failed for handles={handles}: {e}")
    
    def _write_kvs(self, kvs: AdaptedKVCache, start_position: int) -> None:
        assert self._active_cache_tensors_stack, "KV write called outside of use_cache context"
        cache_tensors = self._active_cache_tensors_stack[-1]
        new_kvs = kvs.kvs if hasattr(kvs, "kvs") else kvs  # type: ignore
        assert len(cache_tensors) % 2 == 0 and len(new_kvs) % 2 == 0, "KV lists must be K/V pairs"
        bh, head_dim, new_len = new_kvs[0].shape
        logger.info(f"OFFLOAD: KV write start={start_position}, new_len={new_len}, bh={bh}, d={head_dim}")

        # Keys: cache_key [B,H,D,Lmax], new_key [B*H, D, new_len]
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            B, H, D, Lmax = cache_key.shape
            assert D == head_dim and start_position + new_len <= Lmax
            reshaped = new_key.view(B, H, head_dim, new_len)
            cache_key[:, :, :, start_position:start_position + new_len] = reshaped

        # Values: cache_value [B,H,Lmax,D], new_value [B*H, new_len, D]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            B, H, Lmax, D = cache_value.shape
            assert D == head_dim and start_position + new_len <= Lmax
            reshaped = new_value.view(B, H, new_len, head_dim)
            cache_value[:, :, start_position:start_position + new_len, :] = reshaped

        logger.info("OFFLOAD: KV write finished")

        
        
