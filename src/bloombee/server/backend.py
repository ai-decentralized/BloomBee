from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from time import perf_counter

import torch
import traceback
import numpy as np
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from bloombee.data_structures import InferenceMetadata
from bloombee.server.memory_cache_manager import KVCacheManager
from bloombee.server.task_pool import PrioritizedTaskPool
from bloombee.server.speculative_pruner.pruner_manager import SpeculativePrunerManager
from bloombee.utils.misc import get_size_in_bytes, is_dummy
from bloombee.utils.memory_usage import see_memory_usage
from bloombee.utils.microbatch_config import (
    is_microbatch_enabled,
    get_micro_batch_size,
    get_current_path,
    log_path_entry as mbpipe_log_path_entry,
    MBPIPE_LOG_PREFIX,
)
from bloombee.utils.real_activation_dumper import capture_activation
from pynvml import *
import logging
import hashlib
import time

logger = get_logger(__name__)

# Create dedicated offloading debug logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


def compute_tensor_hash(tensor):
    """Compute SHA256 hash of tensor for debugging - optimized CPU conversion"""
    if tensor is None:
        return "None"
    try:
        # Optimization: Only perform CPU conversion in debug mode to avoid unnecessary performance overhead
        if not getattr(compute_tensor_hash, '_debug_enabled', False):
            return "hash_disabled"  # Disable hash computation in production environment
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    except:
        return "error"

# This flag can be set to enable/disable hash computation
compute_tensor_hash._debug_enabled = False

# def see_memory_usage(message, force=True):
# 	logger = ''
# 	logger += message
# 	nvmlInit()
#  
# 	# nvidia_smi.nvmlInit()
# 	handle = nvmlDeviceGetHandleByIndex(0)
# 	info = nvmlDeviceGetMemoryInfo(handle)
# 	logger += "\n Nvidia-smi: " + str((info.used) / 1024 / 1024 / 1024) + " GB"
# 	
# 	logger += '\n    Memory Allocated: '+str(torch.cuda.memory_allocated() / (1024 * 1024 * 1024)) +'  GigaBytes\n'
# 	logger +=   'Max Memory Allocated: ' + str(
# 		torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)) + '  GigaBytes\n'
# 	print(logger)

class TransformerBackend(ModuleBackend): # hivemind: ModuleBackend.module: nn.Module
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        cache_manager: KVCacheManager,
        pruner_manager: SpeculativePrunerManager,
        is_last_block: bool,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import bloombee.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        # Accept both TensorParallel and our PipelineParallelWrapper
        assert (isinstance(self.module, TensorParallel) or 
                hasattr(self.module, 'devices') and hasattr(self.module, 'module_shards'))
        self.config = config
        self.cache_manager = cache_manager
        self.pruner_manager = pruner_manager
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    self.shard_num_heads.append(submodule.num_heads)
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor(
                    128, dtype=torch.int64
                ), # keep_indices
                BatchTensorDescriptor(
                    1, dtype=torch.int64
                ), # need_pruning
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
                BatchTensorDescriptor(
                    1, 64, 64, dtype=self.dtype
                ), # tree_attention_mask
                BatchTensorDescriptor(
                    128, dtype=torch.int64
                ), #  kv_cache_position_ids
                BatchTensorDescriptor(
                    1, 128, dtype=self.dtype
                ), # draft_tokens
                BatchTensorDescriptor(
                    128, dtype=torch.int64
                ), # prefill_length
                BatchTensorDescriptor(
                    1, dtype=torch.int64
                ), # is_spec_dec
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)
        
        # 🚀 Performance optimization: Pre-allocate position_ids cache
        self._position_ids_cache = {}

        # Decide device placement policy for module based on offloading policy
        offload_policy = cache_manager.offloading_policy
        is_offloading_mode = (
            offload_policy.cache_gpu_percent < 100
            or offload_policy.cache_cpu_percent > 0
            or offload_policy.cache_disk_percent > 0
            or offload_policy.compress_cache
        )

        # 🔧 Note: For offloading mode, we keep the model on GPU
        # The KVCacheManager will handle cache offloading separately
        # Moving model to CPU here causes issues with meta tensors
        # The cache offloading is managed by memory_cache_manager.py

        # Record original devices for restoration when needed (after potential override)
        self.original_devices = self.module.devices
        self.original_output_device_index = getattr(self.module, 'output_device_index', 0)
        self._need_pruning = False
        self._first_get_need_pruning = True
        self._is_spec_decoding = False
        self._is_last_block = is_last_block
        if is_last_block:
            self.module.load_lm_head()
        self._last_keep_indices = None

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            # values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.append(keys)
            # [DEBUG] Log descriptor shape
            logger.debug(f"[MB_DEBUG] get_inference_cache_descriptors: batch_size={batch_size}, num_heads={num_heads}, "
                       f"head_dim={head_dim}, max_length={max_length}, shape={(batch_size, num_heads, head_dim, max_length)}")
        return cache_tensors
    
    def prune_draft_tree(
        self, 
        norm_hidden_states: torch.Tensor, 
        draft_tokens: torch.Tensor, 
        tree_attention_mask: torch.Tensor
    ):
        results = self.pruner_manager.prune_speculation_tree(
            norm_hidden_states,
            draft_tokens,
            tree_attention_mask
        )
        
        keep_indices = results['keep_indices']  # [B, max_keep_len]，padding 为 -1
        # logger.info(f"keep_indices: {keep_indices}")
        self.pruner_manager.middle_keep_indices = keep_indices
        return keep_indices

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            # Before forward, ensure model is on correct device
            self._ensure_model_on_device()
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            # Before backward, ensure model is on correct device
            self._ensure_model_on_device()
            return super().backward(*inputs)

    def _ensure_model_on_device(self):
        """Ensure model is on correct device, load from CPU to GPU if needed"""
        # Optimized: Add fast-path check to avoid repeated device comparison
        if getattr(self, '_device_already_set', False):
            return
        
        # Check if current device differs from original device
        if self.module.devices != self.original_devices:
            # Move model to original devices
            self.module.devices = self.original_devices
            self.module.output_device_index = self.original_output_device_index
            
            # If module has module_shards, move them to original devices
            if hasattr(self.module, 'module_shards'):
                for shard, device in zip(self.module.module_shards, self.original_devices):
                    shard.to(device)
            
            # Mark for delayed initialization
            self.module.need_delayed_init = True
        
        # Mark as already set to skip future checks
        self._device_already_set = True

    @torch.inference_mode() # Enter inference mode, no gradient computation to save memory
    def inference_step( # Each block will execute once
        self,
        hidden_states: torch.Tensor,  # Input hidden state tensor
        hypo_ids: torch.LongTensor,  # Hypothesis IDs
        inference_info: InferenceMetadata,  # Inference-related metadata
    ) -> Tuple[torch.Tensor, ...]:
        try:
            assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]" # Ensure hidden states are 3-dimensional
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            # [ACTIVATION_DUMP] Capture real hidden_states for compression analysis
            # Enabled by: export BLOOMBEE_DUMP_ACTIVATIONS=1
            capture_activation(
                hidden_states,
                block_uid=self.name,
                layer_idx=0,
                inference_info=inference_info
            )
            
            self._ensure_model_on_device()
            
            with self.cache_manager.use_cache(
                *inference_info.cache_handles  # Use cache to reduce memory requirements
            ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter): # Use adapter for inference
                if self._first_get_need_pruning:
                    self._need_pruning = inference_info.need_pruning is not None and inference_info.need_pruning.bool().item()
                    self._is_spec_decoding = inference_info.is_spec_dec is not None and inference_info.is_spec_dec.bool().item()
                    self._first_get_need_pruning = False

                # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
                # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
                # is at least 4-6x less than `autograd_memory`.
                max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info) # Estimate maximum chunk length
                # Debug output removed
                # see_memory_usage("transformer backend inference step : seq_len")
                output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None # Initialize output states
                # print("transformer backend inference step : output_hidden_states", output_hidden_states) # output_hidden_states:None
                # Centralized select: aggregate + reorder + slice
                # [MERGED] Speculative decoding flow with micro-batch support
                kv_cache_position_ids = inference_info.kv_cache_position_ids
                
                logger.debug(f"[MB_DEBUG] backend.inference_step: uid={inference_info.uid}, "
                            f"batch_offset={inference_info.batch_offset}, "
                            f"micro_batch_size={inference_info.micro_batch_size}, "
                            f"full_batch_size={inference_info.full_batch_size}")
                if kv_cache_position_ids is not None and kv_cache_position_ids.numel() > 0:
                    # [Speculative Decoding path] Reorder cache based on position IDs
                    # 1. Get cache for reorder
                    k_pkv_old, v_pkv_old, need_reorder = self.cache_manager.select_cache_for_reorder(
                        kv_cache_position_ids=kv_cache_position_ids
                    )
                    
                    if need_reorder and k_pkv_old is not None:
                        # 2. Reorder and write back, get valid lengths per batch
                        new_prefix_length, kv_valid_lengths = self.cache_manager.reorder_and_write_cache(
                            k_pkv=k_pkv_old,
                            v_pkv=v_pkv_old,
                            kv_cache_position_ids=kv_cache_position_ids,
                        )
                    else:
                        new_prefix_length = inference_info.prefix_length
                        kv_valid_lengths = torch.full((batch_size,), new_prefix_length, device=hidden_states.device)
                    
                    # 3. Select contiguous cache with micro-batch support
                    # [MBPIPE] Pass batch_offset, full_batch_size and micro_batch_size
                    k_pkv, v_pkv, _ = self.cache_manager.select_cache(
                        prefix_length=new_prefix_length,
                        hypo_ids=hypo_ids,
                        kv_cache_position_ids=None,
                        batch_offset=inference_info.batch_offset,
                        full_batch_size=inference_info.full_batch_size,
                        micro_batch_size=inference_info.micro_batch_size,
                    )
                else:
                    # [Standard path] Direct cache selection with micro-batch support
                    # [MBPIPE] Pass batch_offset, full_batch_size and micro_batch_size
                    k_pkv, v_pkv, _ = self.cache_manager.select_cache(
                        prefix_length=inference_info.prefix_length,
                        hypo_ids=hypo_ids,
                        kv_cache_position_ids=None,
                        batch_offset=inference_info.batch_offset,
                        full_batch_size=inference_info.full_batch_size,
                        micro_batch_size=inference_info.micro_batch_size,
                    )
                    new_prefix_length = k_pkv.shape[2] if k_pkv is not None else 0
                    kv_valid_lengths = torch.full((batch_size,), inference_info.prefix_length, device=hidden_states.device)
                
                if k_pkv is not None:
                     logger.debug(f"[MB_DEBUG] Cache selected: k_pkv.shape={k_pkv.shape}")

                layer_past = (k_pkv, v_pkv) if k_pkv is not None else None

                full_mask = None
                device = hidden_states.device
                
                if self._is_spec_decoding:
                    full_mask = self._create_attention_mask(
                        tree_attention_mask=inference_info.tree_attention_mask.to(device),
                        src_len=seq_len + new_prefix_length,
                        past_key_values_length=new_prefix_length,
                        kv_valid_lengths=kv_valid_lengths.to(device),
                        prefill_lengths=inference_info.prefill_length.to(device),
                        device=hidden_states.device,
                    )
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
                if full_mask == None:
                    full_mask = self._create_causal_attention_mask(batch_size, (seq_len + new_prefix_length), new_prefix_length, hidden_states.device)
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None

                for offset in range(0, seq_len, max_chunk_length):
                    hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :]
                    
                    chunk_length = min(max_chunk_length, seq_len - offset)
                    cache_key = (chunk_length, batch_size, hidden_states.device)
                    if cache_key not in self._position_ids_cache:
                        base_ids = torch.arange(0, chunk_length, device=hidden_states.device, dtype=torch.long)
                        self._position_ids_cache[cache_key] = base_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    position_ids = self._position_ids_cache[cache_key] + (new_prefix_length + offset)

                    if self._is_spec_decoding:
                        rotary_position_ids = self._create_tree_position_ids(
                            2, 4, inference_info.prefill_length - 1, kv_valid_lengths, device='cuda'
                        )
                    else:
                        rotary_position_ids = None

                    # logger.info(f"before gather rotary_position_ids: {rotary_position_ids}")
                    # logger.info(f"keep_indices: {inference_info.keep_indices}")
                    # logger.info(f"hidden_states: {hidden_states.shape}")
                    # rotary_position_ids = rotary_position_ids = torch.gather(rotary_position_ids, 1, inference_info.keep_indices.to("cuda")) if rotary_position_ids is not None and inference_info.keep_indices is not None else rotary_position_ids
                    # logger.info(f"after gather rotary_position_ids: {rotary_position_ids}")
                    try:
                        # Fixed: Properly handle forward method return values with position_ids
                        # print(f' About to call module.forward with position_ids...')
                        forward_result = self.module.forward(
                            hidden_states_chunk, 
                            layer_past=layer_past,
                            attention_mask=attention_mask, 
                            use_cache=True,
                            position_ids=position_ids,
                            rotary_position_ids=rotary_position_ids,
                        )
                        
                        if forward_result is None:
                            logger.info(f" ERROR: module.forward returned None!")
                            return (hidden_states, None)
                        
                        output_hidden_states_chunk, new_kvs = forward_result
                        # print(f' Successfully unpacked: output_hidden_states_chunk={output_hidden_states_chunk.shape if output_hidden_states_chunk is not None else None}')
                        
                        # Add forward result debug information
                        # offload_logger.info(f" module.forward completed:")
                        # offload_logger.info(f"   - output_hidden_states_chunk shape: {output_hidden_states_chunk.shape if output_hidden_states_chunk is not None else None}")
                        # offload_logger.info(f"   - new_kvs length: {len(new_kvs) if new_kvs else 0}")
                        # if new_kvs and len(new_kvs) > 0:
                        #     offload_logger.info(f"   - new_kvs[0] shape: {new_kvs[0].shape}")
                        #     offload_logger.info(f"   - new_kvs[0] device: {new_kvs[0].device}")
                        
                    except Exception as e:
                        print(f' ERROR in module.forward: {type(e).__name__}: {e}')
                        import traceback
                        traceback.print_exc()
                        return (hidden_states, None)
                    
                    if seq_len > max_chunk_length:
                        output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk
                    else:
                        output_hidden_states = output_hidden_states_chunk

                # Centralized KV update via KVCacheManager
                # [MERGED] Speculative decoding batched update with micro-batch support
                if self._is_spec_decoding:
                    self.cache_manager.update_cache_batched(new_kvs, kv_valid_lengths)
                else:
                    # [MBPIPE] Pass batch_offset, full_batch_size and micro_batch_size for micro-batch support
                    self.cache_manager.update_cache(
                        new_kvs, new_prefix_length,
                        batch_offset=inference_info.batch_offset,
                        full_batch_size=inference_info.full_batch_size,
                        micro_batch_size=inference_info.micro_batch_size,
                    )

                keep_indices = inference_info.keep_indices
                
                if self._is_spec_decoding and self._need_pruning and self._is_last_block:
                    norm_hidden_states = self.module.rms_norm(output_hidden_states)
                    keep_indices = self.prune_draft_tree(norm_hidden_states, inference_info.draft_tokens, full_mask)
                    
                if self._is_spec_decoding and self._is_last_block:
                    original_hidden_states = output_hidden_states
                    batch_size, seq_len, hidden_size = original_hidden_states.shape
                    device = original_hidden_states.device
                    valid_mask = keep_indices >= 0
                    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(keep_indices)
                    valid_hidden_states = original_hidden_states[batch_idx[valid_mask], keep_indices[valid_mask], :]
                    output_hidden_states = valid_hidden_states.unsqueeze(0)

                self._last_keep_indices = keep_indices + new_prefix_length
                # logger.info(f"update _last_keep_indices: {self._last_keep_indices}")
                
                # In training mode, you need to deploy your whole model in one device and choose a specific middle layer. After saving the middle_states, you can train the MLP network by comparing the middle states and final states logits.
                training_mode = False
                if training_mode and self._is_spec_decoding and inference_info.uid == 'llama-7b-hf.15':
                    norm_hidden_states = self.module.rms_norm(output_hidden_states)
                    self.pruner_manager.middle_states = norm_hidden_states
                
                if training_mode and self._is_spec_decoding and self._is_last_block:
                    norm_hidden_states = self.module.rms_norm(output_hidden_states)
                    final_logits = self.module.lm_head_forward(norm_hidden_states)
                    self.pruner_manager.train_model(final_logits, full_mask, inference_info.draft_tokens)
                
                return (output_hidden_states, keep_indices) # Return output hidden states
                
        except Exception as e:
            logger.exception(
                "inference_step failed for block %s (batch=%s, seq=%s, prefix=%s): %s",
                self.name,
                hidden_states.shape[0],
                hidden_states.shape[1],
                inference_info.prefix_length if 'inference_info' in locals() else None,
                e,
            )
            return (hidden_states, None)  # Return original input as fallback

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def _create_tree_position_ids(
        self, 
        width: int, 
        depth: int, 
        prefill_length: torch.Tensor,  # 每个 batch 样本当前的有效总长度（已包含在该样本在 KV cache 中的位置）
        kv_valid_lengths: torch.Tensor,                  # KV cache 的统一起始偏移量（通常是所有样本对齐后的基准）
        device: torch.device
    ) -> torch.Tensor:
    
        batch_size = prefill_length.shape[0]
        
        # 1. 生成 Tree 模板（相对偏移，根节点为 0）
        tree_position_ids_list = []
        def dfs_generate(node_depth, current_depth):
            tree_position_ids_list.append(node_depth)
            if current_depth < depth:
                for _ in range(width):
                    dfs_generate(node_depth + 1, current_depth + 1)
        dfs_generate(0, 0)
        tree_len = len(tree_position_ids_list)
        tree_position_ids = torch.tensor(tree_position_ids_list, device=device)

        # 判断是否为 Prefill 阶段 (根据输入长度或 past_len 判断)
        # 假设如果 past_len == 0 且输入包含 prefill 部分
        is_prefill = (kv_valid_lengths.max().item() == 0) 

        if is_prefill:
            # --- Prefill 阶段逻辑 ---
            max_prefill_len = prefill_length.max().item()
            total_len = max_prefill_len + tree_len
            full_position_ids = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
            
            for i in range(batch_size):
                pl = prefill_length[i].item()
                # Prefill 部分: [0, 1, ..., pl-1]
                full_position_ids[i, :pl] = torch.arange(pl, device=device)
                # Tree 部分: 接在有效 Prefill 长度 pl 之后，并推到 total_len 的末尾
                full_position_ids[i, max_prefill_len:] = tree_position_ids + pl
                
            return full_position_ids

        else:
            # --- Generation / Verify 阶段逻辑 ---
            # 此时 input_ids 只有 Tree 部分，长度为 tree_len
            full_position_ids = torch.zeros(batch_size, tree_len, dtype=torch.long, device=device)
            
            for i in range(batch_size):
                # 重点：这里的 pl 应该是该样本在上一轮结束时已经确认的 token 总数
                past_len = kv_valid_lengths[i]
                # 这里的 ID 必须累加 past_len 和 pl
                # 如果你的 prefill_length[i] 已经包含了 past_len，则直接加 tree_position_ids
                # 如果 prefill_length[i] 只是当前请求的长度，则需要 past_len + pl + tree_position_ids
                full_position_ids[i, :] = tree_position_ids + past_len
                
            return full_position_ids
    
    def _update_kv_cache_position_ids(self, kv_cache_position_ids, keep_indices):
        if kv_cache_position_ids is None or keep_indices is None:
            return None
        
        if not torch.is_tensor(keep_indices):
            keep_indices = torch.tensor(keep_indices, device=kv_cache_position_ids.device)

        mapping = {int(k.item()): i for i, k in enumerate(keep_indices)}
        new_ids = torch.tensor(
            [mapping.get(int(x.item()), -1) for x in kv_cache_position_ids],
            device=kv_cache_position_ids.device
        )
        return new_ids

    def _create_attention_mask(
        self,
        tree_attention_mask: Optional[torch.Tensor],
        *,
        src_len: int,                # prefix_len + tree_len
        past_key_values_length: int,
        kv_valid_lengths: Optional[torch.Tensor] = None,  # [B] 每个 batch 的有效 KV 长度（后续轮次用）
        prefill_lengths: Optional[torch.Tensor] = None,   # [B] 每个 batch 的实际 prefill 长度（首轮用）
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if tree_attention_mask is None or is_dummy(tree_attention_mask):
            return None
        
        # logger.info(f"tree_attention_mask: {tree_attention_mask.shape}")
        # logger.info(f"src_len: {src_len}")
        # logger.info(f"past_key_values_length: {past_key_values_length}")
        # logger.info(f"kv_valid_lengths: {kv_valid_lengths}")
        # logger.info(f"prefill_lengths: {prefill_lengths}")

        tree_mask = tree_attention_mask
        tree_len = tree_mask.size(1)
        B = tree_mask.size(0)
        prefix_len = src_len - tree_len  # 最大 prefix 长度（包含 padding）
        current_token_count = src_len - past_key_values_length
        
        if current_token_count <= 0:
            return None
        
        if past_key_values_length == 0:
            # ============ 首轮：处理 prefill_lengths ============
            full_mask = torch.zeros(B, src_len, src_len, dtype=torch.bool, device=device)
            
            # 如果没有提供 prefill_lengths，假设所有 batch 的 prefill 长度相同
            if prefill_lengths is None:
                prefill_lengths = torch.full((B,), prefix_len, dtype=torch.long, device=device)
            
            if prefix_len > 0:
                # 位置索引
                row_idx = torch.arange(prefix_len, device=device).view(1, -1, 1)   # [1, prefix_len, 1]
                col_idx = torch.arange(prefix_len, device=device).view(1, 1, -1)   # [1, 1, prefix_len]
                prefill_lens = prefill_lengths.view(B, 1, 1)  # [B, 1, 1]
                
                # causal mask: row >= col
                # 有效 mask: row < prefill_lengths AND col < prefill_lengths
                causal_mask = row_idx >= col_idx  # [1, prefix_len, prefix_len]
                row_valid = row_idx < prefill_lens  # [B, prefix_len, 1]
                col_valid = col_idx < prefill_lens  # [B, 1, prefix_len]
                
                prefix_mask = causal_mask & row_valid & col_valid  # [B, prefix_len, prefix_len]
                full_mask[:, :prefix_len, :prefix_len] = prefix_mask
            
            if prefix_len > 0 and tree_len > 0:
                # tree tokens attend to prefix（只到有效 prefill 位置）
                col_idx = torch.arange(prefix_len, device=device).view(1, 1, -1)  # [1, 1, prefix_len]
                prefill_lens = prefill_lengths.view(B, 1, 1)  # [B, 1, 1]
                col_valid = col_idx < prefill_lens  # [B, 1, prefix_len]
                col_valid = col_valid.expand(B, tree_len, prefix_len)  # [B, tree_len, prefix_len]
                full_mask[:, prefix_len:, :prefix_len] = col_valid

            if tree_len > 0:
                full_mask[:, prefix_len:, prefix_len:] = tree_mask
            
            return full_mask
        
        else:
            # ============ 后续轮次：处理 kv_valid_lengths ============
            current_mask = torch.zeros(B, current_token_count, src_len, dtype=torch.bool, device=device)
            start_pos = past_key_values_length
            
            if start_pos < prefix_len:
                prefix_tokens = min(current_token_count, prefix_len - start_pos)
                for i in range(prefix_tokens):
                    current_mask[:, i, :start_pos + i + 1] = True

                if current_token_count > prefix_tokens:
                    tree_tokens = current_token_count - prefix_tokens
                    current_mask[:, prefix_tokens:, :prefix_len] = True
                    current_mask[:, prefix_tokens:, prefix_len:] = tree_mask[:, :tree_tokens, :]
            else:
                tree_start = start_pos - prefix_len
                if prefix_len > 0:
                    current_mask[:, :, :prefix_len] = True
                current_mask[:, :, prefix_len:] = tree_mask[:, tree_start:tree_start + current_token_count, :]
            
            # 应用 kv_valid_lengths mask
            if kv_valid_lengths is not None:
                current_mask = self._apply_kv_valid_mask(current_mask, kv_valid_lengths, past_key_values_length, device)
            
            return current_mask


    def _apply_kv_valid_mask(
        self,
        mask: torch.Tensor,
        kv_valid_lengths: torch.Tensor,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        将 kv_valid_lengths 应用到 mask 上，屏蔽每个 batch 超出有效长度的 KV 位置
        """
        if kv_len <= 0:
            return mask
        
        B = mask.shape[0]
        key_len = mask.shape[2]
        actual_kv_len = min(kv_len, key_len)
        
        # [1, actual_kv_len]
        kv_positions = torch.arange(actual_kv_len, device=device).unsqueeze(0)
        
        # [B, actual_kv_len] -> [B, 1, actual_kv_len]
        kv_valid_mask = (kv_positions < kv_valid_lengths.unsqueeze(1)).unsqueeze(1)
        
        mask[:, :, :actual_kv_len] = mask[:, :, :actual_kv_len] & kv_valid_mask
        
        return mask
        
    def _create_causal_attention_mask(
        self,
        batch_size: int,
        src_len: int,
        past_key_values_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        B = batch_size
        current_token_count = src_len - past_key_values_length

        if current_token_count <= 0:
            return None
        
        if past_key_values_length == 0:
            full_mask = torch.zeros(B, src_len, src_len, dtype=torch.bool, device=device)
            causal_indices = torch.tril_indices(src_len, src_len, device=device)
            full_mask[:, causal_indices[0], causal_indices[1]] = True
            return full_mask

        current_mask = torch.zeros(B, current_token_count, src_len, dtype=torch.bool, device=device)
        start_pos = past_key_values_length

        for i in range(current_token_count):
            current_mask[:, i, :start_pos + i + 1] = True

        return current_mask

    
    def convert_mask_to_scores(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            raise TypeError(f"Expected bool tensor, got {mask.dtype}")

        scores = torch.full_like(mask, -65504.0, dtype=torch.float)
        scores[mask] = 0.0
        
        return scores

    # Cache writing is centralized in KVCacheManager.update_cache()

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    # print('............... come into the merge_inference_pools_inplace() ' )
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool
        # here, the backend is "blocks" in the server.py line 536

class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends
        self._call_count = 0  # Track number of calls for logging
        # [KVCACHE_OFFLOAD] Track offloaded micro-batch slices: {(mb_offset, mb_size): (k_cpu, v_cpu)}
        self._offloaded_slices: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # Get cache_manager from first backend for offloading operations
        self._cache_manager = next(iter(backends.values())).cache_manager if backends else None
        self._kv_timing_keys = (
            "prefetch_wait_ms",
            "offload_wait_ms",
            "prefetch_wait_calls",
            "offload_wait_calls",
            "prefetch_launch_ms",
            "offload_launch_ms",
            "prefetch_launch_calls",
            "offload_launch_calls",
        )

    def _offload_completed_microbatch(self, k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                       mb_offset: int, mb_size: int, num_heads: int) -> None:
        """
        [KVCACHE_OFFLOAD] Copy micro-batch KV slice from GPU to CPU staging.
        Called after all blocks complete for this micro-batch.
        """
        if k_cache is None or v_cache is None:
            return
        
        # Calculate BH (batch * heads) slice for this micro-batch
        BH_start = mb_offset * num_heads
        BH_end = (mb_offset + mb_size) * num_heads
        
        try:
            # k_cache shape: (S, BH_total, D), v_cache shape: (S, BH_total, D)
            k_slice = k_cache[:, BH_start:BH_end, :].clone()
            v_slice = v_cache[:, BH_start:BH_end, :]
            
            # Async copy to CPU (non-blocking for performance)
            k_cpu = k_slice.to('cpu', non_blocking=True)
            v_cpu = v_slice.to('cpu', non_blocking=True)
            
            # Store in offload tracking dict
            key = (mb_offset, mb_size)
            self._offloaded_slices[key] = (k_cpu, v_cpu)
            
            # Clear GPU slice (optional - helps memory but may not be needed if we just reuse)
            # We don't zero it here as it may be reused in next iteration
            
            offload_logger.info(
                f"[KVCACHE_OFFLOAD] Offloaded: mb_offset={mb_offset}, mb_size={mb_size}, "
                f"BH=[{BH_start}:{BH_end}], k_shape={k_cpu.shape}"
            )
        except Exception as e:
            offload_logger.warning(f"[KVCACHE_OFFLOAD] Offload failed: {e}")

    def _prefetch_if_needed(self, k_cache: torch.Tensor, v_cache: torch.Tensor,
                             mb_offset: int, mb_size: int, num_heads: int) -> None:
        """
        [KVCACHE_OFFLOAD] Copy micro-batch KV slice from CPU back to GPU if previously offloaded.
        Called before processing a micro-batch.
        """
        key = (mb_offset, mb_size)
        if key not in self._offloaded_slices:
            return  # Not offloaded, nothing to prefetch
        
        if k_cache is None or v_cache is None:
            return
            
        k_cpu, v_cpu = self._offloaded_slices[key]
        
        # Calculate BH slice
        BH_start = mb_offset * num_heads
        BH_end = (mb_offset + mb_size) * num_heads
        
        try:
            # Async copy back to GPU
            k_cache[:, BH_start:BH_end, :].copy_(k_cpu.to(k_cache.device, non_blocking=True))
            v_cache[:, BH_start:BH_end, :].copy_(v_cpu.to(v_cache.device, non_blocking=True))
            
            # Remove from tracking
            del self._offloaded_slices[key]
            
            offload_logger.info(
                f"[KVCACHE_OFFLOAD] Prefetched: mb_offset={mb_offset}, mb_size={mb_size}, "
                f"BH=[{BH_start}:{BH_end}]"
            )
        except Exception as e:
            offload_logger.warning(f"[KVCACHE_OFFLOAD] Prefetch failed: {e}")

    def _snapshot_kv_timing(self) -> Optional[Dict[str, float]]:
        if self._cache_manager is None or not hasattr(self._cache_manager, "get_kv_timing_snapshot"):
            return None
        try:
            snapshot = self._cache_manager.get_kv_timing_snapshot(reset=False)
        except Exception as e:
            logger.debug("[KVCACHE_TIMING] runtime snapshot failed: %s", e)
            return None

        normalized: Dict[str, float] = {}
        for key in self._kv_timing_keys:
            try:
                normalized[key] = float(snapshot.get(key, 0.0))
            except Exception:
                normalized[key] = 0.0
        return normalized

    def _compute_kv_timing_delta(
        self, before: Optional[Dict[str, float]], after: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        delta: Dict[str, float] = {}
        for key in self._kv_timing_keys:
            b = 0.0 if before is None else float(before.get(key, 0.0))
            a = 0.0 if after is None else float(after.get(key, 0.0))
            v = max(0.0, a - b)
            if key.endswith("_calls"):
                delta[key] = int(v)
            else:
                delta[key] = v
        delta["_source"] = "runtime_kv_timing"
        delta["_valid"] = 1 if (before is not None and after is not None) else 0
        return delta

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        
        # [MBPIPE] Log current path at _MergedInferenceStep entry (first call only to reduce noise)
        self._call_count += 1
        if self._call_count == 1:
            batch_size = hidden_states.shape[0] if hidden_states.ndim >= 1 else 1
            mbpipe_log_path_entry(logger, "backend._MergedInferenceStep", batch_size=batch_size)
        
        kv_timing_before = self._snapshot_kv_timing()

        # Process all blocks for this micro-batch
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            (hidden_states, keep_indices) = self.backends[inference_info.uid].inference_step(
                hidden_states, hypo_ids, inference_info
            )

        kv_timing_after = self._snapshot_kv_timing()
        kv_timing_delta = self._compute_kv_timing_delta(kv_timing_before, kv_timing_after)

        return (hidden_states, keep_indices, kv_timing_delta)
