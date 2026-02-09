
from dataclasses import dataclass
import contextlib
import asyncio
import torch
import os
import time
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
        
        # [KVCACHE_OFFLOAD] Micro-batch level memory reuse state
        # Since all blocks share one KVCacheManager, staging must be keyed by:
        #   (active_cache_slot_id, micro_batch_index)
        # so each block keeps its own micro-batch history.
        self._mb_cpu_staging = {}  # {(slot_id, mb_index): (k_cpu, v_cpu, prefix_len)}
        self._current_gpu_mb = {}  # {slot_id: mb_index} currently materialized on GPU
        self._offload_enabled = True  # Enable/disable offloading
        # Async KV transfer mode:
        # - explicit override via BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER=0/1
        # - otherwise default to enabled when micro-batching is enabled
        async_kv_env = os.environ.get("BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER")
        if async_kv_env is None:
            microbatch_enabled = os.environ.get("BLOOMBEE_ENABLE_MICROBATCH_PIPELINE", "1") == "1"
            self._async_kv_transfer = microbatch_enabled
            async_mode_source = f"auto(microbatch_enabled={int(microbatch_enabled)})"
        else:
            self._async_kv_transfer = async_kv_env == "1"
            async_mode_source = f"env(BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER={async_kv_env})"
        # Logging control:
        # - default keeps high-frequency KV/offload/prefetch traces at DEBUG
        # - set BLOOMBEE_VERBOSE_KV_LOGS=1 to restore verbose INFO traces
        self._verbose_kv_logs = os.environ.get("BLOOMBEE_VERBOSE_KV_LOGS", "0") == "1"
        self._kv_log_once_keys = set()
        logger.info(
            "[KVCACHE_OFFLOAD] KV transfer mode: %s (%s; set BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER=0/1 to override)",
            "ASYNC" if self._async_kv_transfer else "SYNC",
            async_mode_source,
        )
        logger.info(
            "[KVCACHE_OFFLOAD] KV verbose logs: %s (set BLOOMBEE_VERBOSE_KV_LOGS=1 for per-layer KV traces)",
            "ON" if self._verbose_kv_logs else "OFF",
        )
        
        # [KVCACHE_OFFLOAD] CUDA streams for async data transfer (created lazily)
        # offload_stream: GPU->CPU transfer (runs async while next micro-batch computes)
        # prefetch_stream: CPU->GPU transfer (runs async before compute starts)
        # NOTE: Streams are created lazily to ensure correct device context
        self._offload_stream = None
        self._prefetch_stream = None
        self._streams_device = None  # Track which device streams were created on
        # Per-(slot, mb) transfer completion events for fine-grained synchronization.
        self._mb_offload_events = {}   # {(slot_id, mb_index): torch.cuda.Event}
        self._mb_prefetch_events = {}  # {(slot_id, mb_index): torch.cuda.Event}
        # Track prefetch launched but not yet synchronized into _current_gpu_mb.
        self._pending_gpu_mb = {}      # {slot_id: mb_index}
        # Lightweight KV transfer timing counters (for overlap effectiveness analysis).
        self._enable_kv_wait_timing = os.environ.get("BLOOMBEE_ENABLE_KV_WAIT_TIMING", "1") == "1"
        logger.info(
            "[KVCACHE_OFFLOAD] KV wait timing: %s (set BLOOMBEE_ENABLE_KV_WAIT_TIMING=0 to disable)",
            "ON" if self._enable_kv_wait_timing else "OFF",
        )
        self._kv_timing = {
            "prefetch_wait_ms": 0.0,
            "offload_wait_ms": 0.0,
            "prefetch_wait_calls": 0,
            "offload_wait_calls": 0,
            "prefetch_launch_ms": 0.0,
            "offload_launch_ms": 0.0,
            "prefetch_launch_calls": 0,
            "offload_launch_calls": 0,
        }
        
    def _get_active_cache_slot_id(self) -> Optional[int]:
        """Return a stable identifier for currently active cache tensors."""
        if not self._active_cache_tensors_stack:
            return None
        try:
            cache_tensors = self._active_cache_tensors_stack[-1]
            (k_cache, _), = cache_tensors
            k_data = k_cache.data if hasattr(k_cache, "data") else k_cache
            return id(k_data)
        except Exception:
            return None

    def _get_staging_key(self, mb_index: int) -> Optional[Tuple[int, int]]:
        slot_id = self._get_active_cache_slot_id()
        if slot_id is None:
            return None
        return slot_id, mb_index

    def _compute_microbatch_index(self, batch_offset: int, micro_batch_size: Optional[int] = None) -> int:
        """Compute stable micro-batch index from logical offset."""
        if micro_batch_size is not None and int(micro_batch_size) > 0:
            configured_mb = int(micro_batch_size)
        else:
            configured_mb = max(1, int(getattr(self.offloading_policy, "gpu_batch_size", 1)))
        return max(0, batch_offset // configured_mb)

    def _log_kv_detail(self, message: str, *args):
        """High-frequency KV diagnostics: INFO only when verbose flag is enabled."""
        if self._verbose_kv_logs:
            logger.info(message, *args)
        else:
            logger.debug(message, *args)

    def _log_kv_once(self, key, message: str, *args):
        """Emit an INFO diagnostic once per process for important KV mode/state transitions."""
        if key in self._kv_log_once_keys:
            return
        self._kv_log_once_keys.add(key)
        logger.info(message, *args)

    def _record_kv_timing(self, key: str, delta_ms: float, call_key: Optional[str] = None):
        if not self._enable_kv_wait_timing:
            return
        self._kv_timing[key] = self._kv_timing.get(key, 0.0) + max(0.0, float(delta_ms))
        if call_key is not None:
            self._kv_timing[call_key] = self._kv_timing.get(call_key, 0) + 1

    def get_kv_timing_snapshot(self, reset: bool = False) -> dict:
        """
        Return cumulative KV timing counters used to quantify overlap effectiveness.

        Args:
            reset: If True, clear counters after snapshot.
        """
        snapshot = dict(self._kv_timing)
        if reset:
            for k in list(self._kv_timing.keys()):
                self._kv_timing[k] = 0 if k.endswith("_calls") else 0.0
        return snapshot
        
        
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
        
        # [MBPIPE_KV_VERIFY] High-frequency per-layer read diagnostics
        self._log_kv_detail("[MBPIPE_KV_VERIFY] === KV CACHE READ ===")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache tensor shape: (S={S_full}, BH={BH_full}, D={D})")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache batch capacity: {full_batch_in_cache} (BH={BH_full} / H={H})")
        self._log_kv_detail(
            f"[MBPIPE_KV_VERIFY] Read params: prefix_len={prefix_length}, batch_offset={batch_offset}, "
            f"full_batch={full_batch_size}, micro_batch={micro_batch_size}"
        )
        
        gpu_multiplexing = full_batch_size > 0 and full_batch_in_cache < full_batch_size
        if full_batch_size > 0 and micro_batch_size > 0:
            if gpu_multiplexing:
                mb_index = self._compute_microbatch_index(batch_offset, micro_batch_size)
                slot_id = self._get_active_cache_slot_id()
                current_mb = self._current_gpu_mb.get(slot_id) if slot_id is not None else None
                pending_mb = self._pending_gpu_mb.get(slot_id) if slot_id is not None else None

                # For decode (prefix_length > 0), ensure this micro-batch's KV is on GPU.
                # This must happen inside use_cache context.
                if prefix_length > 0:
                    if current_mb != mb_index:
                        # Avoid relaunching if this micro-batch is already being prefetched.
                        if pending_mb != mb_index:
                            self.prefetch_microbatch_kv(mb_index)
                        self.sync_prefetch(mb_index)
                    elif pending_mb == mb_index:
                        # Prefetch may have been launched earlier; ensure it is complete.
                        self.sync_prefetch(mb_index)

                # GPU multiplexing: all micro-batches map to offset=0.
                actual_mb_size = micro_batch_size if micro_batch_size > 0 else full_batch_in_cache
                BH_offset_start = 0
                BH_offset_end = min(actual_mb_size * H, BH_full)
                BH = BH_offset_end - BH_offset_start
                self._log_kv_once(
                    ("kv_read_mode", "multiplex"),
                    "[MBPIPE_KV_VERIFY] KV READ mode: GPU MULTIPLEXING (micro-batches reuse GPU BH offset=0)",
                )
                self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Mode: GPU MULTIPLEXING - reading BH[0:{BH_offset_end}]")
                self._log_kv_detail(
                    f"[MBPIPE_KV_VERIFY] MATCH CHECK: cache_capacity({full_batch_in_cache}) < full_batch({full_batch_size}) ✓"
                )
            else:
                # Legacy mode: cache holds full batch, use batch_offset for slicing
                BH_offset_start = batch_offset * H
                BH_offset_end = BH_offset_start + micro_batch_size * H
                BH_offset_end = min(BH_offset_end, BH_full)  # Safety clamp
                BH = BH_offset_end - BH_offset_start
                self._log_kv_once(
                    ("kv_read_mode", "legacy"),
                    "[MBPIPE_KV_VERIFY] KV READ mode: LEGACY (cache stores full batch; read uses logical batch_offset)",
                )
                self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Mode: LEGACY - reading BH[{BH_offset_start}:{BH_offset_end}]")
        else:
            # Full batch mode
            BH_offset_start = 0
            BH_offset_end = BH_full
            BH = BH_full
            self._log_kv_once(
                ("kv_read_mode", "full"),
                "[MBPIPE_KV_VERIFY] KV READ mode: FULL BATCH",
            )
            self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Mode: FULL BATCH - reading BH[0:{BH_full}]")

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
            # [KVCACHE_OFFLOAD] Handle TorchMixedDevice reading for micro-batch correctly
            # When cache is MixedDevice and we're reading a BH slice, we need to manually
            # read from each segment and concatenate, because general_copy's cut_indices
            # doesn't work correctly for micro-batch BH offsets
            from bloombee.flexgen_utils.pytorch_backend import DeviceType
            
            if hasattr(k_cache, 'device') and getattr(k_cache.device, 'device_type', None) == DeviceType.MIXED:
                # TorchMixedDevice: manually read from each segment and concatenate
                tensors, seg_points = k_cache.data  # ([gpu_tensor, cpu_tensor, ...], [0, seg1, seg2, ...])
                v_tensors, _ = v_cache.data
                
                s_slice, bh_slice = idx_all[:2]
                bh_start, bh_end = bh_slice.start, bh_slice.stop
                
                self._log_kv_detail(
                    f"[KVCACHE_OFFLOAD] select_cache MixedDevice: seg_points={seg_points}, "
                    f"bh_range=[{bh_start}:{bh_end}], prefix_length={prefix_length}"
                )
                
                k_parts = []
                v_parts = []
                
                for i, (k_seg, v_seg) in enumerate(zip(tensors, v_tensors)):
                    if k_seg is None:
                        continue
                    seg_start = seg_points[i]
                    seg_end = seg_points[i + 1]
                    
                    # Calculate overlap between requested BH range and this segment
                    overlap_start = max(bh_start, seg_start)
                    overlap_end = min(bh_end, seg_end)
                    
                    if overlap_start < overlap_end:
                        # There is overlap, read from this segment
                        # Segment-local indices
                        local_bh_start = overlap_start - seg_start
                        local_bh_end = overlap_end - seg_start
                        
                        logger.debug(f"[KVCACHE_OFFLOAD] select_cache seg[{i}]: "
                                    f"seg_range=[{seg_start}:{seg_end}], local_bh=[{local_bh_start}:{local_bh_end}]")
                        
                        # Get underlying tensor data
                        k_data = k_seg.data if hasattr(k_seg, 'data') else k_seg
                        v_data = v_seg.data if hasattr(v_seg, 'data') else v_seg
                        
                        # Read the slice: (S, local_bh, D)
                        k_part = k_data[s_slice, local_bh_start:local_bh_end, :].to(compute_dst.dev, non_blocking=True)
                        v_part = v_data[s_slice, local_bh_start:local_bh_end, :].to(compute_dst.dev, non_blocking=True)
                        
                        k_parts.append(k_part)
                        v_parts.append(v_part)
                
                # Concatenate all parts along BH dimension
                if len(k_parts) == 1:
                    k_sbh = k_parts[0]
                    v_sbh = v_parts[0]
                else:
                    k_sbh = torch.cat(k_parts, dim=1)
                    v_sbh = torch.cat(v_parts, dim=1)
                    
                logger.debug(f"[KVCACHE_OFFLOAD] select_cache MixedDevice result: k_sbh.shape={k_sbh.shape}")
            else:
                # Non-MixedDevice: use original smart_copy path
                k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
                v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
                k_sbh = _as_torch(k_sel)
                v_sbh = _as_torch(v_sel)
            # logger.info(f\"k_cache: {k_cache.shape}, k_sbh: {k_sbh.shape}\")

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
    
    # ==================== [KVCACHE_OFFLOAD] Micro-batch Offloading ====================
    
    def _ensure_streams_initialized(self, device=None):
        """
        Lazily initialize CUDA streams on the correct device.
        
        This ensures streams are created in the proper device context, avoiding
        CUDA initialization errors on multi-GPU setups.
        """
        if not torch.cuda.is_available():
            return
        if not self._async_kv_transfer:
            return
            
        # Determine device from input or current CUDA context
        if device is None:
            device = torch.cuda.current_device()
        elif hasattr(device, 'index'):
            device = device.index
        elif isinstance(device, torch.device) and device.type == 'cuda':
            device = device.index if device.index is not None else 0
        
        # Create streams if not already created for this device
        if self._offload_stream is None or self._streams_device != device:
            with torch.cuda.device(device):
                self._offload_stream = torch.cuda.Stream()
                self._prefetch_stream = torch.cuda.Stream()
                self._streams_device = device
                logger.debug(f"[KVCACHE_OFFLOAD] Created CUDA streams on device {device}")
    
    def offload_microbatch_kv(self, mb_index: int, prefix_length: int = 0):
        """
        Offload current micro-batch's entire KV cache from GPU to CPU using CUDA stream.
        
        With micro-batch level memory reuse, GPU cache holds ONE micro-batch at a time.
        This function saves the entire GPU cache content to a per-micro-batch CPU buffer
        asynchronously, allowing GPU memory to be reused by the next micro-batch while
        the offload is still in progress.
        
        Args:
            mb_index: Index of the micro-batch (0, 1, 2, ...) to identify which staging buffer
            prefix_length: Sequence length to offload (0 = full sequence)
        """
        self._log_kv_detail(
            f"[MBPIPE_OFFLOAD_DEBUG] offload_microbatch_kv called: mb_index={mb_index}, prefix_length={prefix_length}"
        )
        
        if not self._offload_enabled:
            self._log_kv_detail("[MBPIPE_OFFLOAD_DEBUG] offload DISABLED, returning early")
            return
        staging_key = self._get_staging_key(mb_index)
        if staging_key is None:
            self._log_kv_detail("[MBPIPE_OFFLOAD_DEBUG] No active cache, skipping offload")
            return
        slot_id, _ = staging_key
            
        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        
        try:
            # Get underlying torch tensors
            k_data = k_cache.data if hasattr(k_cache, 'data') else k_cache
            v_data = v_cache.data if hasattr(v_cache, 'data') else v_cache
            
            # [MBPIPE_OFFLOAD_DEBUG] Log the cache shape - THIS IS THE KEY!
            # If k_data.shape[1] == full_batch_size * num_heads, then NO GPU memory savings!
            self._log_kv_detail(
                f"[MBPIPE_OFFLOAD_DEBUG] Cache shape: k_data.shape={k_data.shape if hasattr(k_data, 'shape') else 'N/A'}"
            )
            self._log_kv_detail("[MBPIPE_OFFLOAD_DEBUG] This shape tells us if cache is for FULL batch or MICRO batch")
            
            # Check if data is on GPU
            if hasattr(k_data, 'is_cuda') and not k_data.is_cuda:
                self._log_kv_detail(f"[MBPIPE_OFFLOAD_DEBUG] Cache not on GPU (device={k_data.device}), skipping offload")
                return
            elif hasattr(k_data, 'device') and str(k_data.device) == 'cpu':
                self._log_kv_detail("[MBPIPE_OFFLOAD_DEBUG] Cache on CPU, skipping offload")
                return
            elif isinstance(k_data, tuple):
                # TorchMixedDevice case
                self._log_kv_detail("[MBPIPE_OFFLOAD_DEBUG] Cache is TorchMixedDevice (tuple), skipping standard offload")
                return
            
            # Initialize streams on the correct device
            self._ensure_streams_initialized(k_data.device)

            S_total, BH_mb, D = k_data.shape  # BH_mb = micro_batch_size * num_heads
            
            # [MBPIPE_OFFLOAD_DEBUG] Log the key insight
            H = getattr(self.block_config, "num_attention_heads", 32)
            implied_batch = BH_mb // H
            
            self._log_kv_once(
                ("offload_path", "active"),
                "[KVCACHE_OFFLOAD] micro-batch offload path active (KV snapshots are staged on CPU)",
            )
            self._log_kv_detail(f"[MBPIPE_OFFLOAD] === OFFLOAD MB{mb_index} (slot={slot_id}) ===")
            self._log_kv_detail(f"[MBPIPE_OFFLOAD] GPU cache shape: (S={S_total}, BH={BH_mb}, D={D})")
            self._log_kv_detail(f"[MBPIPE_OFFLOAD] Implied batch in cache: {implied_batch} (BH={BH_mb} / H={H})")
            self._log_kv_detail(f"[MBPIPE_OFFLOAD] Offloading to CPU staging buffer[{slot_id}, {mb_index}]")
            
            # Use full sequence if not specified
            if prefix_length <= 0:
                prefix_length = S_total
            
            # Create or reuse CPU staging buffer for this micro-batch
            if staging_key not in self._mb_cpu_staging:
                k_cpu = torch.empty((S_total, BH_mb, D), dtype=k_data.dtype, device='cpu', pin_memory=True)
                v_cpu = torch.empty((S_total, BH_mb, D), dtype=v_data.dtype, device='cpu', pin_memory=True)
                self._mb_cpu_staging[staging_key] = (k_cpu, v_cpu, 0)  # (k, v, prefix_len)
                self._log_kv_detail(
                    f"[KVCACHE_OFFLOAD] Created CPU staging for slot={slot_id}, mb_index={mb_index}, "
                    f"shape=({S_total}, {BH_mb}, {D})"
                )
            
            k_cpu, v_cpu, _ = self._mb_cpu_staging[staging_key]
            
            # Transfer KV to CPU staging.
            # Default is synchronous for correctness; async mode is optional.
            launch_start = time.perf_counter()
            if self._async_kv_transfer and self._offload_stream is not None:
                # Stream-ordering guard: offload stream waits for the compute stream writes,
                # avoiding a full-device synchronize and preserving overlap potential.
                current_stream = torch.cuda.current_stream(k_data.device)
                with torch.cuda.stream(self._offload_stream):
                    self._offload_stream.wait_stream(current_stream)
                    k_cpu[:prefix_length].copy_(k_data[:prefix_length], non_blocking=True)
                    v_cpu[:prefix_length].copy_(v_data[:prefix_length], non_blocking=True)
                    offload_event = torch.cuda.Event()
                    offload_event.record(self._offload_stream)
                    self._mb_offload_events[staging_key] = offload_event
            else:
                k_cpu[:prefix_length].copy_(k_data[:prefix_length], non_blocking=False)
                v_cpu[:prefix_length].copy_(v_data[:prefix_length], non_blocking=False)
                self._mb_offload_events.pop(staging_key, None)
            self._record_kv_timing(
                "offload_launch_ms",
                (time.perf_counter() - launch_start) * 1000.0,
                "offload_launch_calls",
            )
            
            # Update tracking
            self._mb_cpu_staging[staging_key] = (k_cpu, v_cpu, prefix_length)
            self._current_gpu_mb[slot_id] = None  # This slot no longer has guaranteed active mb
            
            # Calculate memory
            bytes_offloaded = k_data[:prefix_length].numel() * k_data.element_size() * 2
            mb_offloaded = bytes_offloaded / (1024 * 1024)
            
            slot_mbs = sorted(mb for (slot, mb) in self._mb_cpu_staging.keys() if slot == slot_id)
            self._log_kv_detail(
                f"[MBPIPE_OFFLOAD] Offloaded MB{mb_index}: seq_len={prefix_length}, "
                f"size={mb_offloaded:.2f}MB ({'async' if self._async_kv_transfer else 'sync'})"
            )
            self._log_kv_detail(f"[MBPIPE_OFFLOAD] CPU staging buffers for slot={slot_id}: {slot_mbs}")
            
        except Exception as e:
            logger.warning(f"[KVCACHE_OFFLOAD] Offload failed: {e}", exc_info=True)
    
    def prefetch_microbatch_kv(self, mb_index: int):
        """
        Prefetch specified micro-batch's entire KV cache from CPU back to GPU using CUDA stream.
        
        With micro-batch level memory reuse, this restores the specified micro-batch's
        cache content from its CPU staging buffer to the GPU cache asynchronously,
        overwriting whatever was previously in GPU memory.
        
        Call sync_prefetch() to ensure prefetch completes before using the cache.
        
        Args:
            mb_index: Index of the micro-batch (0, 1, 2, ...) to prefetch
        """
        self._log_kv_detail(f"[MBPIPE_PREFETCH] === PREFETCH MB{mb_index} ===")
        
        if not self._offload_enabled:
            self._log_kv_detail("[MBPIPE_PREFETCH] Prefetch DISABLED, skipping")
            return
        
        staging_key = self._get_staging_key(mb_index)
        if staging_key is None:
            logger.debug("[KVCACHE_OFFLOAD] No active cache, skipping prefetch")
            return
        slot_id, _ = staging_key
        slot_mbs = sorted(mb for (slot, mb) in self._mb_cpu_staging.keys() if slot == slot_id)
        self._log_kv_detail(f"[MBPIPE_PREFETCH] CPU staging buffers for slot={slot_id}: {slot_mbs}")
            
        if staging_key not in self._mb_cpu_staging:
            self._log_kv_detail(f"[MBPIPE_PREFETCH] MB{mb_index} NOT in CPU staging - this is EXPECTED for first pass (prefill)")
            self._log_kv_detail("[MBPIPE_PREFETCH] During prefill, no prior KV data exists to prefetch")
            return
            
        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        
        try:
            # Get underlying torch tensors
            k_data = k_cache.data if hasattr(k_cache, 'data') else k_cache
            v_data = v_cache.data if hasattr(v_cache, 'data') else v_cache
            
            # Check if data is on GPU
            if not k_data.is_cuda:
                logger.debug(f"[KVCACHE_OFFLOAD] Cache not on GPU (device={k_data.device}), skipping prefetch")
                return
            
            # Initialize streams on the correct device
            self._ensure_streams_initialized(k_data.device)
            
            # Get CPU staging data
            k_cpu, v_cpu, prefix_length = self._mb_cpu_staging[staging_key]
            
            if prefix_length <= 0:
                logger.warning(f"[KVCACHE_OFFLOAD] mb_index={mb_index} has zero prefix_length, skipping")
                return

            # Transfer KV back to GPU cache.
            # Default is synchronous for correctness; async mode is optional.
            launch_start = time.perf_counter()
            if self._async_kv_transfer and self._prefetch_stream is not None:
                offload_event = self._mb_offload_events.get(staging_key)
                with torch.cuda.stream(self._prefetch_stream):
                    if offload_event is not None:
                        # Wait only for this micro-batch's offload completion (not global stream sync).
                        self._prefetch_stream.wait_event(offload_event)
                    k_data[:prefix_length].copy_(k_cpu[:prefix_length], non_blocking=True)
                    v_data[:prefix_length].copy_(v_cpu[:prefix_length], non_blocking=True)
                    # Record event for synchronization
                    prefetch_event = torch.cuda.Event()
                    prefetch_event.record(self._prefetch_stream)
                    self._mb_prefetch_events[staging_key] = prefetch_event
                # Do not mark as current until sync_prefetch(mb_index) confirms completion.
                self._pending_gpu_mb[slot_id] = mb_index
            else:
                k_data[:prefix_length].copy_(k_cpu[:prefix_length], non_blocking=False)
                v_data[:prefix_length].copy_(v_cpu[:prefix_length], non_blocking=False)
                self._mb_prefetch_events.pop(staging_key, None)
                self._pending_gpu_mb.pop(slot_id, None)
                self._current_gpu_mb[slot_id] = mb_index
            self._record_kv_timing(
                "prefetch_launch_ms",
                (time.perf_counter() - launch_start) * 1000.0,
                "prefetch_launch_calls",
            )
            
            bytes_prefetched = k_cpu[:prefix_length].numel() * k_cpu.element_size() * 2
            mb_prefetched = bytes_prefetched / (1024 * 1024)
            
            self._log_kv_once(
                ("prefetch_path", "active"),
                "[KVCACHE_OFFLOAD] micro-batch prefetch path active (KV snapshots are restored before compute)",
            )
            self._log_kv_detail(
                f"[MBPIPE_PREFETCH] Prefetched MB{mb_index}: seq_len={prefix_length}, "
                f"size={mb_prefetched:.2f}MB ({'async' if self._async_kv_transfer else 'sync'})"
            )
            self._log_kv_detail(f"[MBPIPE_PREFETCH] GPU cache slot={slot_id} now contains MB{mb_index} data")
            
        except Exception as e:
            logger.warning(f"[KVCACHE_OFFLOAD] Prefetch failed: {e}", exc_info=True)
    
    def sync_prefetch(self, mb_index: Optional[int] = None):
        """
        Wait for async prefetch to complete.
        
        Call this after prefetch_microbatch_kv() and before using the cache
        to ensure the CPU->GPU transfer has finished.

        Args:
            mb_index: Optional micro-batch index. If provided, synchronize only this
                      micro-batch staging key for the current active slot.
        """
        if not self._async_kv_transfer:
            return

        if mb_index is None:
            if self._mb_prefetch_events:
                for event in self._mb_prefetch_events.values():
                    wait_start = time.perf_counter()
                    event.synchronize()
                    self._record_kv_timing(
                        "prefetch_wait_ms",
                        (time.perf_counter() - wait_start) * 1000.0,
                        "prefetch_wait_calls",
                    )
                self._mb_prefetch_events.clear()
            if self._pending_gpu_mb:
                for slot_id, pending_mb in self._pending_gpu_mb.items():
                    self._current_gpu_mb[slot_id] = pending_mb
                self._pending_gpu_mb.clear()
            logger.debug("[KVCACHE_OFFLOAD] Prefetch sync complete (all)")
            return

        staging_key = self._get_staging_key(mb_index)
        if staging_key is not None:
            event = self._mb_prefetch_events.pop(staging_key, None)
            if event is not None:
                wait_start = time.perf_counter()
                event.synchronize()
                self._record_kv_timing(
                    "prefetch_wait_ms",
                    (time.perf_counter() - wait_start) * 1000.0,
                    "prefetch_wait_calls",
                )
            slot_id, _ = staging_key
            if self._pending_gpu_mb.get(slot_id) == mb_index:
                self._current_gpu_mb[slot_id] = mb_index
                self._pending_gpu_mb.pop(slot_id, None)
            logger.debug("[KVCACHE_OFFLOAD] Prefetch sync complete for mb_index=%s", mb_index)
    
    def sync_offload(self, mb_index: Optional[int] = None):
        """
        Wait for async offload to complete.
        
        Call this to ensure GPU->CPU transfer has finished before freeing GPU memory
        or before prefetching a new micro-batch that would overwrite the cache.

        Args:
            mb_index: Optional micro-batch index. If provided, synchronize only this
                      micro-batch staging key for the current active slot.
        """
        if not self._async_kv_transfer:
            return

        if mb_index is not None:
            staging_key = self._get_staging_key(mb_index)
            if staging_key is not None:
                event = self._mb_offload_events.get(staging_key)
                if event is not None:
                    wait_start = time.perf_counter()
                    event.synchronize()
                    self._record_kv_timing(
                        "offload_wait_ms",
                        (time.perf_counter() - wait_start) * 1000.0,
                        "offload_wait_calls",
                    )
                    logger.debug("[KVCACHE_OFFLOAD] Offload sync complete for mb_index=%s", mb_index)
            return

        if self._offload_stream is not None and self._streams_device is not None:
            try:
                with torch.cuda.device(self._streams_device):
                    wait_start = time.perf_counter()
                    self._offload_stream.synchronize()
                self._record_kv_timing(
                    "offload_wait_ms",
                    (time.perf_counter() - wait_start) * 1000.0,
                    "offload_wait_calls",
                )
                logger.debug("[KVCACHE_OFFLOAD] Offload sync complete")
            except Exception as e:
                logger.warning(f"[KVCACHE_OFFLOAD] Offload sync failed: {e}")
    
    def clear_offload_state(self):
        """Clear all offload tracking state and free CPU staging buffers."""
        # Ensure any pending offload completes (only if streams were used)
        if self._async_kv_transfer and self._offload_stream is not None and self._mb_cpu_staging:
            self.sync_offload()
        num_cleared = len(self._mb_cpu_staging)
        self._mb_cpu_staging.clear()
        self._current_gpu_mb.clear()
        self._pending_gpu_mb.clear()
        self._mb_offload_events.clear()
        self._mb_prefetch_events.clear()
        if num_cleared > 0:
            logger.info(f"[KVCACHE_OFFLOAD] Cleared offload state: {num_cleared} staged micro-batches")
    
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
        
        # [MBPIPE_WRITE_DEBUG] Log the cache shape and write parameters
        H = getattr(self.block_config, "num_attention_heads", 32)
        cache_batch_size = BH_dst // H
        
        # [MBPIPE_MULTIPLEX] Detect GPU multiplexing mode by actual cache capacity.
        # Multiplexing is active when allocated cache capacity is smaller than logical full batch.
        gpu_multiplexing = (full_batch_size > 0 and cache_batch_size < full_batch_size)
        if micro_batch_size > 0:
            policy_mb = max(1, int(getattr(self.offloading_policy, "gpu_batch_size", 1)))
            if policy_mb != int(micro_batch_size):
                logger.warning(
                    "[MBPIPE_KV_VERIFY] micro_batch_size mismatch: "
                    f"policy.gpu_batch_size={policy_mb}, runtime_micro_batch_size={micro_batch_size}. "
                    "Using runtime value for mb_index mapping."
                )
        
        # [MBPIPE_KV_VERIFY] High-frequency per-layer write diagnostics
        self._log_kv_detail("[MBPIPE_KV_VERIFY] === KV CACHE WRITE ===")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache tensor shape: (S={S_total}, BH={BH_dst}, D={D_dst})")
        self._log_kv_detail(f"[MBPIPE_KV_VERIFY] Cache batch capacity: {cache_batch_size} (BH={BH_dst} / H={H})")
        self._log_kv_detail(
            f"[MBPIPE_KV_VERIFY] Write params: start_pos={start_position}, batch_offset={batch_offset}, "
            f"full_batch={full_batch_size}, micro_batch={micro_batch_size}"
        )

        if gpu_multiplexing:
            self._log_kv_once(
                ("kv_write_mode", "multiplex"),
                "[MBPIPE_KV_VERIFY] KV WRITE mode: GPU MULTIPLEXING (all micro-batches write to GPU BH offset=0)",
            )
            self._log_kv_detail(
                f"[MBPIPE_KV_VERIFY] Mode: GPU MULTIPLEXING (cache={cache_batch_size} < full_batch={full_batch_size})"
            )
            self._log_kv_detail("[MBPIPE_KV_VERIFY] All micro-batches write to offset=0, reusing same GPU slots")
        elif full_batch_size > 0:
            self._log_kv_once(
                ("kv_write_mode", "legacy"),
                "[MBPIPE_KV_VERIFY] KV WRITE mode: LEGACY (cache stores full batch; write uses logical batch_offset)",
            )
            self._log_kv_detail(
                f"[MBPIPE_KV_VERIFY] Mode: LEGACY (cache={cache_batch_size}, full_batch={full_batch_size})"
            )
        else:
            self._log_kv_once(
                ("kv_write_mode", "single"),
                "[MBPIPE_KV_VERIFY] KV WRITE mode: SINGLE BATCH",
            )
            self._log_kv_detail("[MBPIPE_KV_VERIFY] Mode: SINGLE BATCH (no micro-batching)")
        

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
        
        if full_batch_size > 0:
            # Micro-batch mode: BH_src is current micro-batch, BH_dst is cache capacity.
            if gpu_multiplexing:
                # Multiplexing mode: always write to offset 0.
                BH_offset_start = 0
                BH_offset_end = BH_src
                self._log_kv_detail(
                    f"[MBPIPE_MULTIPLEX] _write_kvs: GPU multiplexing ACTIVE, writing to [0:{BH_src}] "
                    f"(all micro-batches reuse same GPU slots)"
                )
            else:
                # Legacy mode: cache holds full batch, use batch_offset for slicing
                BH_offset_start = batch_offset * H
                BH_offset_end = BH_offset_start + BH_src
                if BH_offset_end > BH_dst:
                    logger.error(f"[MBPIPE_DEBUG] BH offset out of bounds: {BH_offset_end} > {BH_dst}, "
                                f"clamping to BH_dst={BH_dst}")
                    BH_offset_end = BH_dst
                logger.debug(f"[MBPIPE] _write_kvs LEGACY: BH_slice=[{BH_offset_start}:{BH_offset_end}]")
        else:
            # Full batch mode: if BH_src < BH_dst, auto-adapt by writing to first BH_src entries
            # This handles the case where request batch_size < server cache batch_size
            if BH_src <= BH_dst:
                BH_offset_start = 0
                BH_offset_end = BH_src
                if BH_src < BH_dst:
                    actual_batch_src = BH_src // H
                    actual_batch_dst = BH_dst // H
                    self._log_kv_once(
                        ("kv_write_mode", "auto_adapt"),
                        "[MBPIPE_DEBUG] KV auto-adapt mode observed (request batch smaller than cache capacity)",
                    )
                    self._log_kv_detail(
                        f"[MBPIPE_DEBUG] Auto-adapting: writing first {BH_src} of {BH_dst} BH entries "
                        f"(batch {actual_batch_src} of {actual_batch_dst})"
                    )
                logger.debug(f"[MBPIPE] _write_kvs FULL: BH_slice=[0:{BH_src}]")
            else:
                # This should not happen - source is larger than destination
                raise AssertionError(f"BH mismatch: src {BH_src} > dst {BH_dst}, cannot write")

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

        # [KVCACHE_OFFLOAD] Handle TorchMixedDevice (GPU+CPU split) properly for micro-batch slicing
        # The issue: general_copy's cut_indices uses segment boundaries on BOTH src and dst,
        # but for micro-batch, src size != dst slice size (e.g., src=128 but dst_idx spans [128:256])
        from bloombee.flexgen_utils.pytorch_backend import DeviceType
        
        def _write_to_cache(cache, src_tt, dst_idx, cache_name):
            """Write source tensor to cache, handling TorchMixedDevice segment splitting."""
            if hasattr(cache, 'device') and getattr(cache.device, 'device_type', None) == DeviceType.MIXED:
                # TorchMixedDevice: manually split writes across segments
                tensors, seg_points = cache.data  # ([gpu_tensor, cpu_tensor, ...], [0, seg1, seg2, ...])
                s_slice, bh_slice, d_slice = dst_idx
                bh_start, bh_end = bh_slice.start, bh_slice.stop
                
                logger.debug(f"[KVCACHE_OFFLOAD] {cache_name} is MixedDevice, seg_points={seg_points}, "
                            f"bh_range=[{bh_start}:{bh_end}]")
                
                # Track position in source tensor
                src_offset = 0
                
                for i, seg_tensor in enumerate(tensors):
                    if seg_tensor is None:
                        continue
                    seg_start = seg_points[i]
                    seg_end = seg_points[i + 1]
                    
                    # Calculate overlap between dst BH range and this segment
                    overlap_start = max(bh_start, seg_start)
                    overlap_end = min(bh_end, seg_end)
                    
                    if overlap_start < overlap_end:
                        # There is overlap, need to write to this segment
                        overlap_len = overlap_end - overlap_start
                        
                        # Source: take next overlap_len elements
                        src_slice_start = src_offset
                        src_slice_end = src_offset + overlap_len
                        
                        # Destination: offset within this segment
                        dst_bh_start = overlap_start - seg_start
                        dst_bh_end = dst_bh_start + overlap_len
                        
                        logger.debug(f"[KVCACHE_OFFLOAD] {cache_name} seg[{i}]: src_bh=[{src_slice_start}:{src_slice_end}], "
                                    f"dst_bh=[{dst_bh_start}:{dst_bh_end}]")
                        
                        # Slice source data
                        src_data = src_tt.data[:, src_slice_start:src_slice_end, :]
                        
                        # Get destination tensor data and write
                        if hasattr(seg_tensor, 'data'):
                            dst_data = seg_tensor.data[s_slice, dst_bh_start:dst_bh_end, d_slice]
                            dst_data.copy_(src_data, non_blocking=True)
                        else:
                            seg_tensor[s_slice, dst_bh_start:dst_bh_end, d_slice].copy_(src_data, non_blocking=True)
                        
                        src_offset += overlap_len
            else:
                # Non-MixedDevice: use regular general_copy
                general_copy(cache, dst_idx, src_tt, None)
        
        # Actual write with MixedDevice handling
        _write_to_cache(k_cache, k_src_tt, dst_idx, "k_cache")
        _write_to_cache(v_cache, v_src_tt, dst_idx, "v_cache")

        # In GPU multiplexing mode, immediately offload this micro-batch cache snapshot
        # while still inside use_cache context, so the next micro-batch can reuse GPU slots.
        if gpu_multiplexing:
            mb_index = self._compute_microbatch_index(batch_offset, micro_batch_size)
            self.offload_microbatch_kv(mb_index, prefix_length=end_position)
    
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
        
    def update_cache_batched(
        self,
        new_kvs: AdaptedKVCache,
        kv_valid_lengths: torch.Tensor,
    ) -> None:
        """
        Batch speculative decoding 专用：每个 batch 从不同位置写入 KV cache
        """
        # 快速路径：所有 batch 的 start_position 相同
        if (kv_valid_lengths == kv_valid_lengths[0]).all():
            self._write_kvs(new_kvs, kv_valid_lengths[0].item())
            return
        
        # 慢速路径：逐 batch 写入
        assert self._active_cache_tensors_stack, "write called outside of use_cache context"
        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        S_total, BH_dst, D_dst = k_cache.shape
        
        new_kvs_data = new_kvs.kvs if hasattr(new_kvs, "kvs") else new_kvs
        key, value = new_kvs_data
        
        def _to_torch(x):
            if hasattr(x, 'device') and (
                getattr(getattr(x, 'device', None), 'device_type', None) == DeviceType.COMPRESSED
                or (hasattr(x, 'data') and isinstance(getattr(x, 'data'), tuple) and len(getattr(x, 'data')) == 3)
            ):
                return x.device.decompress(x)
            return getattr(x, 'data', x)
        
        key_t = _to_torch(key)       # (B*H, D, s_new)
        value_t = _to_torch(value)   # (B*H, s_new, D)
        
        BH_src, D_src, s_new = key_t.shape
        H = getattr(self.block_config, "num_attention_heads", None)
        B = BH_src // H
        
        if key_t.dtype != k_cache.dtype:
            key_t = key_t.to(dtype=k_cache.dtype)
        if value_t.dtype != v_cache.dtype:
            value_t = value_t.to(dtype=v_cache.dtype)
        
        # (B*H, D, s_new) -> (s_new, B*H, D)
        k_write = key_t.permute(2, 0, 1)
        v_write = value_t.permute(1, 0, 2)
        
        for i in range(B):
            start_pos = kv_valid_lengths[i].item()
            end_pos = min(start_pos + s_new, S_total)
            actual_len = end_pos - start_pos
            
            if actual_len <= 0:
                continue
            
            head_start = i * H
            head_end = (i + 1) * H
            
            # 提取第 i 个 batch 的数据并写入
            k_batch = k_write[:actual_len, head_start:head_end, :].contiguous()
            v_batch = v_write[:actual_len, head_start:head_end, :].contiguous()
            
            dst_idx = (slice(start_pos, end_pos), slice(head_start, head_end), slice(0, D_src))
            
            k_src_tt = TorchTensor.create_from_torch(k_batch, self.attention_compute)
            v_src_tt = TorchTensor.create_from_torch(v_batch, self.attention_compute)
            
            general_copy(k_cache, dst_idx, k_src_tt, None)
            general_copy(v_cache, dst_idx, v_src_tt, None)
        
    def reorder_and_write_cache(
        self,
        k_pkv: torch.Tensor,
        v_pkv: torch.Tensor,
        kv_cache_position_ids: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """
        将分散的 positions 重排到连续位置 [0, N)
        
        Args:
            k_pkv, v_pkv: [B, H, S_old, D] 原始取出的 cache
            kv_cache_position_ids: [B, pos_len] 每个 batch 需要保留的 positions（-1 为无效）
        
        Returns:
            max_new_length: 重排后的最大有效长度（用于 select_cache）
            valid_lengths: [B] 每个 batch 的实际有效长度（用于 attention mask）
        """
        assert self._active_cache_tensors_stack, "write called outside of use_cache"
    
        B, H, S_old, D = k_pkv.shape
        device = k_pkv.device
        
        if kv_cache_position_ids.dim() == 1:
            kv_cache_position_ids = kv_cache_position_ids.unsqueeze(0)
        
        # 计算每个 batch 的完整索引（prefix + valid tree positions）
        batch_all_positions = []
        valid_lengths_list = []
        
        for i in range(B):
            positions = kv_cache_position_ids[i]
            valid_mask = positions >= 0
            valid_positions = positions[valid_mask].tolist()
            
            if len(valid_positions) > 0:
                # root_position 是第一个有效位置
                root_position = valid_positions[0]
                # prefix: [0, 1, ..., root_position - 1]
                prefix_positions = list(range(root_position))
                # 完整序列: prefix + valid tree positions
                all_positions = prefix_positions + valid_positions
            else:
                # 没有有效的 tree positions，保持原样
                all_positions = list(range(S_old))
            
            batch_all_positions.append(all_positions)
            valid_lengths_list.append(len(all_positions))
        
        valid_lengths = torch.tensor(valid_lengths_list, dtype=torch.long, device=device)
        max_new_length = valid_lengths.max().item()
        
        # 创建新的 reordered cache
        k_new = torch.zeros(B, H, max_new_length, D, dtype=k_pkv.dtype, device=device)
        v_new = torch.zeros(B, H, max_new_length, D, dtype=v_pkv.dtype, device=device)
        
        for i in range(B):
            all_positions = batch_all_positions[i]
            valid_len = len(all_positions)
            
            if valid_len > 0:
                positions_tensor = torch.tensor(all_positions, dtype=torch.long, device=device)
                # 从原 cache 的 all_positions 位置取出，写入新 cache 的 [0, valid_len)
                k_new[i, :, :valid_len, :] = k_pkv[i, :, positions_tensor, :]
                v_new[i, :, :valid_len, :] = v_pkv[i, :, positions_tensor, :]
        
        # 写回 cache（从 position 0 开始）
        self.write_pkv_cache(k_new, v_new, start_position=0)
        
        return max_new_length, valid_lengths
    
    def select_cache_for_reorder(
        self,
        kv_cache_position_ids: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        为 reorder 准备：取出所有 batch 需要的 positions 的并集
        """
        assert self._active_cache_tensors_stack, "select_cache called outside of use_cache"
        
        cache_tensors = self._active_cache_tensors_stack[-1]
        (k_cache, v_cache), = cache_tensors
        S_full, BH, D = k_cache.shape
        
        compute_dst = self.attention_compute
        
        H = getattr(self.block_config, "num_attention_heads", None)
        B = BH // H
        
        def _as_torch(x):
            return x.data if hasattr(x, "data") else x
        
        if kv_cache_position_ids.dim() == 1:
            kv_cache_position_ids = kv_cache_position_ids.unsqueeze(0)
        
        # 找出需要的最大 position
        valid_mask = kv_cache_position_ids >= 0
        if not valid_mask.any():
            return None, None, False
        
        max_position = kv_cache_position_ids[valid_mask].max().item()
        
        # 取 [0, max_position] 范围
        prefix_length = int(max_position) + 1
        idx_all = (slice(0, prefix_length), slice(0, BH))
        
        k_sel, _ = k_cache.smart_copy(compute_dst, idx_all)
        v_sel, _ = v_cache.smart_copy(compute_dst, idx_all)
        k_sbh = _as_torch(k_sel)
        v_sbh = _as_torch(v_sel)
        
        def _to_pkv(x_sbh: torch.Tensor) -> torch.Tensor:
            return x_sbh.view(prefix_length, B, H, D).permute(1, 2, 0, 3)
        
        k_pkv = _to_pkv(k_sbh)
        v_pkv = _to_pkv(v_sbh)
        
        # 判断是否需要 reorder
        need_reorder = True  # 只要有 kv_cache_position_ids 就需要
        
        return k_pkv, v_pkv, need_reorder
