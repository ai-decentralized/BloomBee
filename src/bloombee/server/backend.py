from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
                kv_cache_position_ids = inference_info.kv_cache_position_ids
                # logger.info(f"_last_keep_indices: {self._last_keep_indices}")
                # logger.info(f"keep_indices: {inference_info.keep_indices}")
                # logger.info(f"before format kv_cache_position_ids: {kv_cache_position_ids}")
                # if self._is_spec_decoding and not self._need_pruning:
                #     kv_cache_position_ids = self._update_kv_cache_position_ids(kv_cache_position_ids, self._last_keep_indices)
                # logger.info(f"after format kv_cache_position_ids: {kv_cache_position_ids}")
                if kv_cache_position_ids is not None and kv_cache_position_ids.numel() > 0:
                    # 1. 取出需要 reorder 的 cache
                    k_pkv_old, v_pkv_old, need_reorder = self.cache_manager.select_cache_for_reorder(
                        kv_cache_position_ids=kv_cache_position_ids
                    )
                    
                    if need_reorder and k_pkv_old is not None:
                        # 2. 重排并写回，获取每个 batch 的有效长度
                        new_prefix_length, kv_valid_lengths = self.cache_manager.reorder_and_write_cache(
                            k_pkv=k_pkv_old,
                            v_pkv=v_pkv_old,
                            kv_cache_position_ids=kv_cache_position_ids,
                        )
                    else:
                        new_prefix_length = inference_info.prefix_length
                        kv_valid_lengths = torch.full((batch_size,), new_prefix_length, device=hidden_states.device)
                    
                    # 3. 取连续的 cache
                    k_pkv, v_pkv, _ = self.cache_manager.select_cache(
                        prefix_length=new_prefix_length,
                        hypo_ids=hypo_ids,
                        kv_cache_position_ids=None,
                    )
                else:
                    k_pkv, v_pkv, _ = self.cache_manager.select_cache(
                        prefix_length=inference_info.prefix_length,
                        hypo_ids=hypo_ids,
                        kv_cache_position_ids=None
                    )
                    new_prefix_length = k_pkv.shape[2] if k_pkv is not None else 0
                    kv_valid_lengths = torch.full((batch_size,), inference_info.prefix_length, device=hidden_states.device)

                layer_past = (k_pkv, v_pkv) if k_pkv is not None else None
                # if k_pkv is not None:
                #     logger.info(f"kv cache size: {k_pkv.shape}, prefix_length: {inference_info.prefix_length}")
                # else:
                #     logger.info(f"kv cache is None, prefix_length: {inference_info.prefix_length}")

                # logger.info(f"past_key_values_length: {new_prefix_length}, hidden_states.device: {hidden_states.device}")
                # logger.info(f"inference_info.tree_attention_mask, shape: {inference_info.tree_attention_mask.shape}")
                full_mask = None
                device = hidden_states.device
                # logger.info(f"kv_valid_lengths: {kv_valid_lengths}")
                
                if self._is_spec_decoding:
                    full_mask = self._create_attention_mask(
                        tree_attention_mask=inference_info.tree_attention_mask.to(device),
                        src_len=seq_len + new_prefix_length,
                        past_key_values_length=new_prefix_length,
                        kv_valid_lengths=kv_valid_lengths.to(device),  # 新增参数
                        prefill_lengths=inference_info.prefill_length.to(device),
                        device=hidden_states.device,
                    )
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
                if full_mask == None:
                    full_mask = self._create_causal_attention_mask(batch_size, (seq_len + new_prefix_length), new_prefix_length, hidden_states.device)
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
                # logger.info(f"full_mask, shape: {full_mask.shape},  {full_mask}")
                # logger.info(f"hidden states in backend before compute: {hidden_states}")
                for offset in range(0, seq_len, max_chunk_length): # Iterate through sequence to process hidden states in chunks   only run offset=0
                    hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :] # Get current hidden states chunk
                    # print('transformer backend inference step() offset ', offset )
                    # print('transformer backend inference step() offset + max_chunk_length',  (offset + max_chunk_length))
                    
                    #  Generate correct position_ids for this chunk
                    chunk_length = min(max_chunk_length, seq_len - offset)
                    # Optimized: Reuse cached position_ids base tensor
                    cache_key = (chunk_length, batch_size, hidden_states.device)
                    if cache_key not in self._position_ids_cache:
                        # Create base position_ids (0 to chunk_length-1)
                        base_ids = torch.arange(0, chunk_length, device=hidden_states.device, dtype=torch.long)
                        self._position_ids_cache[cache_key] = base_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    # Add offset to cached base tensor (avoids creating new tensor)
                    position_ids = self._position_ids_cache[cache_key] + (new_prefix_length + offset)
                    # logger.info(f"position_ids: {position_ids}")
                    # logger.info(f"prefill_length: {inference_info.prefill_length}, kv_valid_lengths: {kv_valid_lengths}")
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
                            use_cache=True,  #  Keep use_cache=True to get cache tensors
                            position_ids=position_ids,  #  Pass the generated position_ids
                            rotary_position_ids=rotary_position_ids,
                        )
                        # print(f' module.forward returned: {type(forward_result)}, length: {len(forward_result) if forward_result else "None"}')
                        
                        if forward_result is None:
                            logger.info(f" ERROR: module.forward returned None!")
                            return (hidden_states, None)  # Return original input as fallback
                        
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
                        return (hidden_states, None)  # Return original input as fallback
                    
                    if seq_len > max_chunk_length:
                        output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk # Store output
                    else:
                        output_hidden_states = output_hidden_states_chunk  # saves one memcopy # Copy memory only once
                    # layer_past = new_kvs # Update cache state

                # logger.info(f"inference_step, output_hidden_states: {output_hidden_states}")
                # Centralized KV update via KVCacheManager (logs OFFLOAD: KV write ...)
                if self._is_spec_decoding:
                    self.cache_manager.update_cache_batched(new_kvs, kv_valid_lengths)
                else:
                    self.cache_manager.update_cache(new_kvs, new_prefix_length)
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
        # print('............... come into the _MergedInferenceStep __call__' )
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            # print('............... come into the _MergedInferenceStep __call__ inference_info.uid ', inference_info.uid)
            (hidden_states, keep_indices) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        # import pdb; pdb.set_trace()
        return (hidden_states, keep_indices)