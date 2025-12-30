
from dataclasses import dataclass
import contextlib
import asyncio
import torch
import os
from typing import Optional, Tuple, AsyncContextManager, Sequence

from bloombee.server.memory_cache import MemoryCache, AdaptedKVCache, KVCacheMetadata
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import DeviceType, TorchDisk, TorchMixedDevice, TorchTensor, general_copy
from hivemind.utils import TensorDescriptor, enter_asynchronously, get_logger

from bloombee.data_structures import Handle
from bloombee.utils.asyncio import shield_and_wait
from bloombee.utils.misc import get_size_in_bytes

from transformers import PretrainedConfig

logger = get_logger(__name__)


class KVCacheManager:
    def __init__(self, cache_max_size_tokens: int, 
                 max_alloc_timeout: int, 
                 policy: Policy, 
                 env: ExecutionEnv,
                 block_config: PretrainedConfig):
        # Initialize as 2D array structure
        self.env = env
        self.runtime_pid = os.getpid()
        self.device = self.get_cache_device(policy)
        self.cache = MemoryCache(cache_max_size_tokens, max_alloc_timeout, policy, block_config, self.device)
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
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert all(descr.device is not None for descr in descriptors), "please specify allocated devices"
        if self.max_alloc_timeout is not None and timeout is not None:
            timeout = min(timeout, self.max_alloc_timeout)

        allocation_tokens = self.get_allocation_size_tokens(*descriptors)
        allocation_size = allocation_tokens * self.size_per_token() * len(descriptors)

        gib = 1024**3
        cur_tokens, max_tokens = self.current_size_tokens, self.max_size_tokens
        max_size = max_tokens * self.size_per_token() * len(descriptors)
        cur_size = cur_tokens * self.size_per_token() * len(descriptors)
        # logger.info(f"size_per_token: {self.size_per_token()}")
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        used_pct = (cur_size / max_size * 100.0) if max_size != 0 and max_size != 2**64 - 1 else 0.0
        # logger.info(
        #     f"rpc_inference.wait_for_alloc(size={allocation_size / gib:.2f} GiB), "
        #     f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({used_pct:.1f}%)"
        # )

        alloc_task = asyncio.create_task(self.cache._schedule_alloc(allocation_tokens, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            # logger.info(f"rpc_inference.alloc_done(size={allocation_size / gib:.2f} GiB)")
            yield handles
        finally:
            self.cache._free(allocation_tokens, alloc_task)
            
            
    @staticmethod
    def get_allocation_size_tokens(*descriptors: TensorDescriptor) -> int:
        allocation_tokens_num = 0
        for descr in descriptors:
            allocation_tokens_num = max(descr.shape[-1], allocation_tokens_num) 
        return allocation_tokens_num
        
    
    def add_cache(self, kvs: AdaptedKVCache, start_position: int):
        self._write_kvs(kvs, start_position)
                
    def update_cache(
        self, new_kvs: AdaptedKVCache, start_position: int,
        batch_offset: int = 0, full_batch_size: int = 0, micro_batch_size: int = 0
    ):
        """
        Update KV cache with new values.
        
        Args:
            new_kvs: New KV tensors to write
            start_position: Start position along sequence dimension
            batch_offset: Start index in the full batch for micro-batch (default 0)
            full_batch_size: Total batch size for micro-batch support (0 = no micro-batch)
            micro_batch_size: Actual size of this micro-batch (0 = use full_batch_size - batch_offset)
        """
        self._write_kvs(new_kvs, start_position, batch_offset, full_batch_size, micro_batch_size)
    
    def tokens_left(self) -> int:
        return self.cache.tokens_left

    @property
    def current_size_tokens(self) -> int:
        return self.cache.current_size_tokens

    @property
    def max_size_tokens(self) -> int:
        return self.cache.max_size_tokens
    
    def size_per_token(self) -> int:
        cache_values_per_block = 2 * self.block_config.hidden_size
        cache_values_per_block //= self.block_config.num_key_value_groups
        return cache_values_per_block * get_size_in_bytes(torch.float16)
    
    def select_cache(
        self,
        prefix_length: int,
        hypo_ids: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        batch_offset: int = 0,
        full_batch_size: int = 0,
        micro_batch_size: int = 0,
    ):
        """
        Return standard KV for computation
        K, V: torch.Tensor, both with shape (B, H, S, D), located on compute_dst (CPU or GPU)
        Convention:
        - Internal cache is stored along dimension (S, B*H, D)
        - If mixed device (MIXED), segments will be merged on compute_dst and returned
        
        Args:
            prefix_length: Length of prefix to read from cache
            hypo_ids: Optional hypothesis IDs for reordering
            kv_cache_position_ids: Optional position IDs for cache
            batch_offset: Start index in the full batch for micro-batch (default 0)
            full_batch_size: Total batch size (0 = no micro-batch, read entire cache)
            micro_batch_size: Actual size of this micro-batch (0 = use full_batch_size - batch_offset)
        """
        assert self._active_cache_tensors_stack, "select_cache called outside of use_cache"
        if prefix_length <= 0:
            return None, None, False

        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        S_full, BH_full, D = k_cache.shape
        assert prefix_length <= S_full, f"prefix_length={prefix_length} > seq_len={S_full}"
        
        # [MBPIPE] Debug log - use DEBUG level to avoid log spam
        logger.debug(f"[MBPIPE] select_cache: prefix_length={prefix_length}, "
                    f"cache_shape=(S={S_full}, BH={BH_full}, D={D}), "
                    f"batch_offset={batch_offset}, full_batch_size={full_batch_size}, micro_batch_size={micro_batch_size}")
        
        # Micro-batch support: compute BH slice
        H = getattr(self.block_config, "num_attention_heads", BH_full)
        full_batch_in_cache = BH_full // H  # Actual batch size in allocated cache
        
        if full_batch_size > 0 and micro_batch_size > 0:
            # Micro-batch mode: read only the slice for this micro-batch
            # Use explicit micro_batch_size from caller
            BH_offset_start = batch_offset * H
            BH_offset_end = BH_offset_start + micro_batch_size * H
            BH_offset_end = min(BH_offset_end, BH_full)  # Safety clamp
            BH = BH_offset_end - BH_offset_start
            logger.debug(f"[MBPIPE] select_cache MICRO: H={H}, micro_batch_size={micro_batch_size}, "
                        f"BH_slice=[{BH_offset_start}:{BH_offset_end}]")
        else:
            # Full batch mode
            BH_offset_start = 0
            BH_offset_end = BH_full
            BH = BH_full
            logger.debug(f"[MBPIPE] select_cache FULL: H={H}, BH_slice=[0:{BH_full}]")

        # Target device for computation (CPU/GPU)
        compute_dst = self.attention_compute  # 统一在计算设备上物化

        # Path determination (whether MIXED)
        if self.offloading_policy.cpu_cache_compute and (
            self.device.device_type == DeviceType.MIXED and getattr(k_cache.data[0][0], "shape", None) is not None
        ):
            path = 2
        else:
            path = 0 if not self.offloading_policy.cpu_cache_compute else 1

        # Required slice - MUST include BH slice for micro-batch support
        need_reorder = False
        if kv_cache_position_ids is None or kv_cache_position_ids.numel() == 0:
            idx_all = (slice(0, prefix_length), slice(BH_offset_start, BH_offset_end))
        else:
            root_position = kv_cache_position_ids[0]
            prefix_positions = list(range(root_position))  # [0, 1, 2, ..., root-1]
            s_indices = prefix_positions + kv_cache_position_ids.tolist()  # 完整序列
            expected_continuous = list(range(len(s_indices)))
            need_reorder = False if (s_indices == expected_continuous) else True
            prefix_length = len(s_indices)
            # [MBPIPE FIX] Include BH slice for micro-batch support
            idx_all = (s_indices, slice(BH_offset_start, BH_offset_end))
            logger.debug(f"[MBPIPE] select_cache with kv_cache_position_ids: s_len={len(s_indices)}, "
                        f"BH_slice=[{BH_offset_start}:{BH_offset_end}]")

        # Utility: get underlying torch.Tensor
        def _as_torch(x):
            return x.data if hasattr(x, "data") else x

        # 1) Materialize to (S, BH, D) torch.Tensor (located on compute_dst)
        if path == 0:
            # Directly slice prefix on compute_dst (view for same device, copy/decompress for cross-device)
            k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
            v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
            k_sbh = _as_torch(k_sel)
            v_sbh = _as_torch(v_sel)
            # logger.info(f"k_cache: {k_cache.shape}, k_sbh: {k_sbh.shape}")

        elif path == 1:
            # Use compute_dst workspace to carry (S, BH, D)
            k_buf, v_buf = compute_dst.next_attention_compute_workspace()
            general_copy(k_buf, idx_all, k_cache, idx_all)
            general_copy(v_buf, idx_all, v_cache, idx_all)
            k_sbh, v_sbh = _as_torch(k_buf), _as_torch(v_buf)

        else:  # path == 2, MIXED: GPU segment + other segments merged to compute_dst
            gpu_k_part = k_cache.data[0][0][:prefix_length]  # (S, BH_gpu, D)
            gpu_v_part = v_cache.data[0][0][:prefix_length]
            BH_gpu = int(gpu_k_part.shape[1])

            # Copy remaining segments to compute_dst workspace
            k_rest, v_rest = compute_dst.next_attention_compute_workspace()
            idx_rest = (slice(0, prefix_length), slice(BH_gpu, BH))
            general_copy(k_rest, idx_rest, k_cache, idx_rest)
            general_copy(v_rest, idx_rest, v_cache, idx_rest)
            k_rest_t, v_rest_t = _as_torch(k_rest), _as_torch(v_rest)

            # If compute_dst is not on GPU, need to move GPU segment to compute_dst then concatenate
            if gpu_k_part.device != k_rest_t.device:
                gpu_k_part = gpu_k_part.to(k_rest_t.device, non_blocking=True)
                gpu_v_part = gpu_v_part.to(v_rest_t.device, non_blocking=True)

            k_sbh = torch.cat([gpu_k_part, k_rest_t[:, BH_gpu:BH, :]], dim=1)
            v_sbh = torch.cat([gpu_v_part, v_rest_t[:, BH_gpu:BH, :]], dim=1)

        # 2) (S, BH, D) -> (B, H, S, D) standard PKV view (zero-copy)
        H = getattr(self.block_config, "num_attention_heads", None)
        assert H is not None, "block_config.num_attention_heads is required"
        assert (k_sbh.shape[1] % H) == 0, f"BH={k_sbh.shape[1]} not divisible by H={H}"
        B = k_sbh.shape[1] // H

        def _to_pkv(x_sbh: torch.Tensor) -> torch.Tensor:
            # (S, BH, D) -> (S, B, H, D) -> (B, H, S, D)
            return x_sbh.view(prefix_length, B, H, D).permute(1, 2, 0, 3)

        k_pkv = _to_pkv(k_sbh)
        v_pkv = _to_pkv(v_sbh)
        
        # [MBPIPE] Debug log output shape
        logger.debug(f"[MBPIPE] select_cache OUTPUT: k_pkv={k_pkv.shape}, B={B}, S={prefix_length}")

        # Optional: reorder batch by hypo_ids
        # if hypo_ids is not None:
        #     # hypo_ids: shape (B,)
        #     k_pkv = k_pkv.index_select(0, hypo_ids)
        #     v_pkv = v_pkv.index_select(0, hypo_ids)
        return k_pkv, v_pkv, need_reorder


    
    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        with self.cache.use_cache(*handles) as cache_tensors:
            # Keep underlying tensors in the stack for centralized writes,
            # but yield clones to callers to prevent accidental in-place edits
            # logger.info(f"use cache, cache_tensors: {cache_tensors}, len={len(cache_tensors)}")
            self._active_cache_tensors_stack.append(cache_tensors)
            try:
                # safe_views = tuple(t.detach().clone() for t in cache_tensors)
                yield cache_tensors
            finally:
                self._active_cache_tensors_stack.pop()

    def delete_cache(self, *handles: Handle):
        """Explicitly delete cache handles to free space early."""
        try:
            self.cache.force_free(*handles)
        except Exception as e:
            logger.warning(f"OFFLOAD: delete_cache failed for handles={handles}: {e}")
    
    def _write_kvs(self, kvs, start_position: int, batch_offset: int = 0, full_batch_size: int = 0, micro_batch_size: int = 0) -> None:
        """
        Write new_kvs to current active cache:
        - Target cache_tensors: k_cache, v_cache, both with shape (S_total, B*H, D)
        - Write start position: start_position (along sequence dimension)
        - Source new_kvs:
            key:   (BH_micro, D, s_new) for micro-batch or (BH_full, D, s_new) for full batch
            value: (BH_micro, s_new, D) for micro-batch or (BH_full, s_new, D) for full batch
        - batch_offset: Start index in the full batch for micro-batch (along B dimension)
        - full_batch_size: Total batch size (0 = no micro-batch, use entire cache)
        - micro_batch_size: Actual size of this micro-batch (0 = use full_batch_size - batch_offset)
        """
        assert self._active_cache_tensors_stack, "KV write called outside of use_cache context"
        cache_tensors = self._active_cache_tensors_stack[-1]  # TorchTensor
        (k_cache, v_cache), = cache_tensors
        S_total, BH_dst, D_dst = k_cache.shape
        
        # [MBPIPE] Debug log - use DEBUG level
        logger.debug(f"[MBPIPE] _write_kvs: start_pos={start_position}, cache_shape=(S={S_total}, BH={BH_dst}), "
                    f"batch_offset={batch_offset}, micro_batch_size={micro_batch_size}")

        # Extract (key, value)
        new_kvs = kvs.kvs if hasattr(kvs, "kvs") else kvs
        key, value = new_kvs

        # If possibly FlexGen wrapper/compression, convert to torch.Tensor
        try:
            from bloombee.flexgen_utils.pytorch_backend import DeviceType
            def _to_torch(x):
                if hasattr(x, 'device') and (
                    getattr(getattr(x, 'device', None), 'device_type', None) == DeviceType.COMPRESSED
                    or (hasattr(x, 'data') and isinstance(getattr(x, 'data'), tuple) and len(getattr(x, 'data')) == 3)
                ):
                    return x.device.decompress(x)
                return getattr(x, 'data', x)
        except Exception:
            def _to_torch(x):
                return getattr(x, 'data', x)

        key_t = _to_torch(key)       # (BH_micro, D, s_new)
        value_t = _to_torch(value)   # (BH_micro, s_new, D)

        # Shape validation
        assert key_t.ndim == 3 and value_t.ndim == 3, f"new_kvs dims invalid: key {key_t.shape}, value {value_t.shape}"
        BH_src, D_src, s_new = key_t.shape
        assert value_t.shape == (BH_src, s_new, D_src), f"value shape {value_t.shape} != (BH, s_new, D)"
        assert D_src == D_dst, f"D mismatch: src {D_src} vs dst {D_dst}"
        
        # [MBPIPE] Debug log source shape
        logger.debug(f"[MBPIPE] _write_kvs SRC: key={key_t.shape}, BH_src={BH_src}, s_new={s_new}")

        # Micro-batch support: compute BH offset for batch slicing
        H = getattr(self.block_config, "num_attention_heads", BH_dst)
        if full_batch_size > 0:
            # Micro-batch mode: BH_src is for micro-batch, BH_dst is for full batch
            micro_batch_size = BH_src // H
            BH_offset_start = batch_offset * H
            BH_offset_end = BH_offset_start + BH_src
            assert BH_offset_end <= BH_dst, f"BH offset out of bounds: {BH_offset_end} > {BH_dst}"
            logger.debug(f"[MBPIPE] _write_kvs MICRO: BH_slice=[{BH_offset_start}:{BH_offset_end}]")
        else:
            # Full batch mode: BH_src should match BH_dst
            assert BH_src == BH_dst, f"BH mismatch: src {BH_src} vs dst {BH_dst}"
            BH_offset_start = 0
            BH_offset_end = BH_src
            logger.debug(f"[MBPIPE] _write_kvs FULL: BH_slice=[0:{BH_src}]")

        end_position = start_position + s_new
        if not (0 <= start_position < S_total and end_position <= S_total):
            # Out of bounds: use overwrite-tail policy
            key_t = key_t[:, :, -S_total:]
            value_t = value_t[:, -S_total:, :]
            s_new = S_total
            start_position = 0
            end_position = S_total

        # Align dtype
        if key_t.dtype != k_cache.dtype:
            key_t = key_t.to(dtype=k_cache.dtype)
        if value_t.dtype != v_cache.dtype:
            value_t = value_t.to(dtype=v_cache.dtype)

        # Transform to internal layout (s_new, BH, D)
        k_write = key_t.permute(2, 0, 1)   # (s_new, BH_src, D)
        v_write = value_t.permute(1, 0, 2) # (s_new, BH_src, D)

        # Target slice - now with BH offset for micro-batch support
        dst_idx = (slice(start_position, start_position + s_new), 
                   slice(BH_offset_start, BH_offset_end), 
                   slice(0, D_src))
        
        # [MBPIPE] Debug log write target
        logger.debug(f"[MBPIPE] _write_kvs TARGET: S=[{start_position}:{start_position + s_new}], "
                    f"BH=[{BH_offset_start}:{BH_offset_end}]")

        # Wrap source into TorchTensor
        k_src_tt = TorchTensor.create_from_torch(k_write, self.attention_compute)
        v_src_tt = TorchTensor.create_from_torch(v_write, self.attention_compute)

        # Actual write
        general_copy(k_cache, dst_idx, k_src_tt, None)
        general_copy(v_cache, dst_idx, v_src_tt, None)
    
    def write_pkv_cache(self, k_pkv: torch.Tensor, v_pkv: torch.Tensor, start_position: int = 0) -> None:
        assert self._active_cache_tensors_stack, "write_pkv_cache called outside of use_cache context"
        
        B, H, S, D = k_pkv.shape
        BH = B * H
        k_write = k_pkv.reshape(BH, S, D).permute(0, 2, 1)  # (B, H, S, D) -> (B*H, S, D) -> (B*H, D, S)
        v_write = v_pkv.reshape(BH, S, D)                    # (B, H, S, D) -> (B*H, S, D)
        self._write_kvs(
            kvs=(k_write, v_write),
            start_position=start_position
        )