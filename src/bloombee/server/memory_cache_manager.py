
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
        # 初始化为二维数组结构
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
        allocation_size = allocation_tokens * self.size_per_token()

        gib = 1024**3
        cur_tokens, max_tokens = self.current_size_tokens, self.max_size_tokens
        max_size = max_tokens * self.size_per_token()
        cur_size = cur_tokens * self.size_per_token()
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        used_pct = (cur_size / max_size * 100.0) if max_size != 0 and max_size != 2**64 - 1 else 0.0
        logger.info(
            f"rpc_inference.wait_for_alloc(size={allocation_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({used_pct:.1f}%)"
        )

        alloc_task = asyncio.create_task(self.cache._schedule_alloc(allocation_tokens, *descriptors, timeout=timeout))
        try:
            handles = await shield_and_wait(alloc_task)
            logger.info(f"rpc_inference.alloc_done(size={allocation_size / gib:.2f} GiB)")
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
        self, new_kvs: AdaptedKVCache, start_position: int
    ):
        self._write_kvs(new_kvs, start_position)
    
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
    ):
        """
        返回用于计算的标准 KV
        K, V: torch.Tensor, 形状均为 (B, H, S, D)，位于 compute_dst（CPU或GPU）上
        约定：
        - 内部缓存沿维度 (S, B*H, D) 存储
        - 若为混合设备(MIXED)则会在 compute_dst 上合并段并返回
        """
        assert self._active_cache_tensors_stack, "select_cache called outside of use_cache"
        if prefix_length <= 0:
            return None

        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        S_full, BH, D = k_cache.shape
        assert prefix_length <= S_full, f"prefix_length={prefix_length} > seq_len={S_full}"

        # 计算发生的目标设备（CPU/GPU）
        compute_dst = self.attention_compute  # 统一在计算设备上物化

        # 路径判定（是否 MIXED）
        if self.offloading_policy.cpu_cache_compute and (
            self.device.device_type == DeviceType.MIXED and getattr(k_cache.data[0][0], "shape", None) is not None
        ):
            path = 2
        else:
            path = 0 if not self.offloading_policy.cpu_cache_compute else 1

        # 需要的切片
        idx_all = (slice(0, prefix_length), slice(0, BH))

        # 工具：拿到底层 torch.Tensor
        def _as_torch(x):
            return x.data if hasattr(x, "data") else x

        # 1) 物化到 (S, BH, D) 的 torch.Tensor（位于 compute_dst）
        if path == 0:
            # 直接在 compute_dst 上切前缀（同设备为视图，跨设备为复制/解压）
            k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
            v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
            k_sbh = _as_torch(k_sel)
            v_sbh = _as_torch(v_sel)
            logger.info(f"k_cache: {k_cache.shape}, k_sbh: {k_sbh.shape}")

        elif path == 1:
            # 使用 compute_dst 的 workspace 承载 (S, BH, D)
            k_buf, v_buf = compute_dst.next_attention_compute_workspace()
            general_copy(k_buf, idx_all, k_cache, idx_all)
            general_copy(v_buf, idx_all, v_cache, idx_all)
            k_sbh, v_sbh = _as_torch(k_buf), _as_torch(v_buf)

        else:  # path == 2, MIXED: GPU 段 + 其他段合并到 compute_dst
            gpu_k_part = k_cache.data[0][0][:prefix_length]  # (S, BH_gpu, D)
            gpu_v_part = v_cache.data[0][0][:prefix_length]
            BH_gpu = int(gpu_k_part.shape[1])

            # 其余段复制到 compute_dst 的 workspace
            k_rest, v_rest = compute_dst.next_attention_compute_workspace()
            idx_rest = (slice(0, prefix_length), slice(BH_gpu, BH))
            general_copy(k_rest, idx_rest, k_cache, idx_rest)
            general_copy(v_rest, idx_rest, v_cache, idx_rest)
            k_rest_t, v_rest_t = _as_torch(k_rest), _as_torch(v_rest)

            # 若 compute_dst 不在 GPU，需要把 GPU 段搬到 compute_dst 再拼
            if gpu_k_part.device != k_rest_t.device:
                gpu_k_part = gpu_k_part.to(k_rest_t.device, non_blocking=True)
                gpu_v_part = gpu_v_part.to(v_rest_t.device, non_blocking=True)

            k_sbh = torch.cat([gpu_k_part, k_rest_t[:, BH_gpu:BH, :]], dim=1)
            v_sbh = torch.cat([gpu_v_part, v_rest_t[:, BH_gpu:BH, :]], dim=1)

        # 2) (S, BH, D) -> (B, H, S, D) 的标准 PKV 视图（零拷贝）
        H = getattr(self.block_config, "num_attention_heads", None)
        assert H is not None, "block_config.num_attention_heads is required"
        assert (k_sbh.shape[1] % H) == 0, f"BH={k_sbh.shape[1]} not divisible by H={H}"
        B = k_sbh.shape[1] // H

        def _to_pkv(x_sbh: torch.Tensor) -> torch.Tensor:
            # (S, BH, D) -> (S, B, H, D) -> (B, H, S, D)
            return x_sbh.view(prefix_length, B, H, D).permute(1, 2, 0, 3)

        k_pkv = _to_pkv(k_sbh)
        v_pkv = _to_pkv(v_sbh)

        # 可选：按 hypo_ids 重排 batch
        # if hypo_ids is not None:
        #     # hypo_ids: shape (B,)
        #     k_pkv = k_pkv.index_select(0, hypo_ids)
        #     v_pkv = v_pkv.index_select(0, hypo_ids)
        return k_pkv, v_pkv


    
    @contextlib.contextmanager
    def use_cache(self, *handles: Handle) -> Sequence[torch.Tensor]:
        with self.cache.use_cache(*handles) as cache_tensors:
            # Keep underlying tensors in the stack for centralized writes,
            # but yield clones to callers to prevent accidental in-place edits
            logger.info(f"use cache, cache_tensors: {cache_tensors}, len={len(cache_tensors)}")
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
    
    def _write_kvs(self, kvs, start_position: int) -> None:
        """
        将 new_kvs 写入当前活跃 cache：
        - 目标 cache_tensors：k_cache, v_cache，形状均为 (S_total, B*H, D)
        - 写入起点：start_position（沿 sequence 维）
        - 源 new_kvs：
            key:   (B*H, D, s_new)
            value: (B*H, s_new, D)
        """
        assert self._active_cache_tensors_stack, "KV write called outside of use_cache context"
        cache_tensors = self._active_cache_tensors_stack[-1]  # TorchTensor
        (k_cache, v_cache), = cache_tensors
        logger.info(f"_active_cache_tensors_stack, k_cache: {k_cache}")
        S_total, BH_dst, D_dst = k_cache.shape

        # 取出 (key, value)
        new_kvs = kvs.kvs if hasattr(kvs, "kvs") else kvs
        key, value = new_kvs

        # 若可能是 FlexGen 的封装/压缩，转成 torch.Tensor（与你给的 to_torch_tensor 一致的逻辑）
        try:
            from bloombee.flexgen_utils.pytorch_backend import DeviceType
            def _to_torch(x):
                if hasattr(x, 'device') and (
                    getattr(getattr(x, 'device', None), 'device_type', None) == DeviceType.COMPRESSED
                    or (hasattr(x, 'data') and isinstance(getattr(x, 'data'), tuple) and len(getattr(x, 'data')) == 3)
                ):
                    return x.device.decompress(x)  # 解压到 torch.Tensor
                return getattr(x, 'data', x)      # TorchTensor -> torch.Tensor, 否则原样返回
        except Exception:
            def _to_torch(x):
                return getattr(x, 'data', x)

        key_t = _to_torch(key)       # (BH, D, s_new)
        value_t = _to_torch(value)   # (BH, s_new, D)

        # 形状与范围校验
        assert key_t.ndim == 3 and value_t.ndim == 3, f"new_kvs dims invalid: key {key_t.shape}, value {value_t.shape}"
        BH_src, D_src, s_new = key_t.shape
        assert value_t.shape == (BH_src, s_new, D_src), f"value shape {value_t.shape} != (BH, s_new, D)"
        assert BH_src == BH_dst, f"BH mismatch: src {BH_src} vs dst {BH_dst}"
        assert D_src == D_dst, f"D mismatch: src {D_src} vs dst {D_dst}"

        end_position = start_position + s_new
        assert 0 <= start_position < S_total and end_position <= S_total, \
            f"write range [{start_position}, {end_position}) out of bounds (S_total={S_total})"

        # 可选：对齐 dtype（以目标 cache 为准）
        if key_t.dtype != k_cache.dtype:
            key_t = key_t.to(dtype=k_cache.dtype)
        if value_t.dtype != v_cache.dtype:
            value_t = value_t.to(dtype=v_cache.dtype)

        # 仅做视图变换到内部布局 (s_new, BH, D)；不产生新内存
        k_write = key_t.permute(2, 0, 1)   # (s_new, BH, D)
        v_write = value_t.permute(1, 0, 2) # (s_new, BH, D)

        # 目标切片
        dst_idx = (slice(start_position, start_position + s_new), slice(0, BH_src), slice(0, D_src))

        # 将源 torch.Tensor 包装成 TorchTensor（device 用 compute 设备即可；底层断言已关，允许不一致）
        # 这样 general_copy 能统一处理各类设备/压缩/分段
        k_src_tt = TorchTensor.create_from_torch(k_write, self.attention_compute)
        v_src_tt = TorchTensor.create_from_torch(v_write, self.attention_compute)

        logger.info(f"_write_kvs, dst_idx: {dst_idx}")

        # 真正写入（兼容 COMPRESSED / MIXED / DISK 等）
        general_copy(k_cache, dst_idx, k_src_tt, None)
        general_copy(v_cache, dst_idx, v_src_tt, None)


        
        