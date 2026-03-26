from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from time import perf_counter

import torch
import traceback
import numpy as np
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
from bloombee.utils.hivemind_compat import BatchTensorDescriptor, TensorDescriptor
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
import threading

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
            # IMPORTANT:
            # Flex decode path (`mha_gen_llama`) uses attention head count when building
            # K/V updates (BH = batch * num_attention_heads). KV cache descriptors must
            # match that contract, otherwise we hit BH mismatches (e.g. 32 vs 128).
            # So keep shard attention heads here; do NOT downscale by key-value groups.
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
            
            # t0 = time.perf_counter()
            with self.cache_manager.use_cache(
                *inference_info.cache_handles  # Use cache to reduce memory requirements
            ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter): # Use adapter for inference
                def _flag_to_bool(value) -> bool:
                    if value is None:
                        return False
                    if torch.is_tensor(value):
                        if value.numel() == 0:
                            return False
                        return bool(value.bool().any().item())
                    return bool(value)

                # Parse flags per request (not just first-ever call), otherwise spec/non-spec
                # mode can get stuck after the first request served by this backend.
                self._need_pruning = _flag_to_bool(inference_info.need_pruning)
                self._is_spec_decoding = _flag_to_bool(inference_info.is_spec_dec)

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
                    k_pkv, v_pkv, cache_len = self.cache_manager.select_cache_without_reorder(
                        kv_cache_position_ids, 
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
                    cache_len = k_pkv.shape[2] if k_pkv is not None else 0
                    
                # t2 = time.perf_counter()
                # logger.info(f"inference_step: cache reorder (if needed) and selection took {t2 - t1:.4f} seconds")

                layer_past = (k_pkv, v_pkv) if k_pkv is not None else None

                full_mask = None
                device = hidden_states.device
                
                if self._is_spec_decoding:
                    full_mask = inference_info.tree_attention_mask.to(device)
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
                if full_mask == None:
                    full_mask = self._create_causal_attention_mask(batch_size, (seq_len + cache_len), cache_len, hidden_states.device)
                    attention_mask = self.convert_mask_to_scores(full_mask) if full_mask is not None else None
                    
                for offset in range(0, seq_len, max_chunk_length): # Iterate through sequence to process hidden states in chunks   only run offset=0
                    hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :] # Get current hidden states chunk
                    # print('transformer backend inference step() offset ', offset )
                    # print('transformer backend inference step() offset + max_chunk_length',  (offset + max_chunk_length))
                    
                    chunk_length = min(max_chunk_length, seq_len - offset)
                    cache_key = (chunk_length, batch_size, hidden_states.device)
                    if cache_key not in self._position_ids_cache:
                        base_ids = torch.arange(0, chunk_length, device=hidden_states.device, dtype=torch.long)
                        self._position_ids_cache[cache_key] = base_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    # Add offset to cached base tensor (avoids creating new tensor)
                    position_ids = self._position_ids_cache[cache_key] + (cache_len + offset)
                    if self._is_spec_decoding:
                        rotary_position_ids = self._create_tree_position_ids_with_invalid_cache(
                            width=2,
                            depth=3,
                            prefill_length=inference_info.prefill_length - 1,
                            kv_cache_position_ids=kv_cache_position_ids,
                            batch_offset=inference_info.batch_offset,
                            device="cuda",
                            target_seq_len=seq_len)
                    else:
                        rotary_position_ids = None
                    
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
                        
                        # t5 = time.perf_counter()
                        # logger.info(f"inference_step: module.forward call took {t5 - t4:.4f} seconds")
                        
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
                        logger.error(f'ERROR in module.forward: {type(e).__name__}: {e}')
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
                    # self.cache_manager.update_cache_batched(new_kvs, kv_valid_lengths)
                    self.cache_manager.update_cache_and_async_reorder(
                        new_kvs, 
                        self._slice_batch_aligned(kv_cache_position_ids, inference_info.batch_offset, inference_info.batch_offset + inference_info.micro_batch_size, inference_info.full_batch_size),
                        batch_offset=inference_info.batch_offset,
                        full_batch_size=inference_info.full_batch_size,
                        micro_batch_size=inference_info.micro_batch_size,
                        cache_tensors=cache_tensors,)
                else:
                    self.cache_manager.update_cache(
                        new_kvs, 
                        cache_len,
                        batch_offset=inference_info.batch_offset,
                        full_batch_size=inference_info.full_batch_size,
                        micro_batch_size=inference_info.micro_batch_size,) 
                    
                keep_indices = self._normalize_keep_indices(
                    inference_info.keep_indices,
                    batch_size=output_hidden_states.shape[0],
                    seq_len=output_hidden_states.shape[1],
                    device=output_hidden_states.device,
                )
                
                
                # logger.info(f"inference_step: KV cache update took {t6 - t5:.4f} seconds")
                
                # In training mode, you need to deploy your whole model in one device and choose a specific middle layer. After saving the middle_states, you can train the MLP network by comparing the middle states and final states logits.
                training_mode = False
                if training_mode and self._is_spec_decoding and inference_info.uid == 'llama-7b-hf.15':
                    self.pruner_manager.middle_states = output_hidden_states
                
                training_model_mode = False
                if training_mode and training_model_mode and self._is_spec_decoding and self._is_last_block:
                    norm_hidden_states = self.module.rms_norm(output_hidden_states)
                    final_logits = self.module.lm_head_forward(norm_hidden_states)
                    middle_norm_hidden_states = self.module.rms_norm(self.pruner_manager.middle_states)
                    self.pruner_manager.train_model(middle_norm_hidden_states, final_logits, full_mask, inference_info.draft_tokens)
                    
                training_lm_head_mode = False
                if training_mode and training_lm_head_mode and self._is_spec_decoding and self._is_last_block:
                    norm_hidden_states = self.module.rms_norm(output_hidden_states)
                    middle_norm_hidden_states = self.module.rms_norm(self.pruner_manager.middle_states)
                    self.pruner_manager.train_lm_head(middle_norm_hidden_states, norm_hidden_states)
                
                if not training_mode and self._is_spec_decoding and self._need_pruning and self._is_last_block:
                    # norm_hidden_states = self.module.rms_norm(output_hidden_states)
                    # keep_indices = self.prune_draft_tree(norm_hidden_states, inference_info.draft_tokens, full_mask)
                    keep_indices = keep_indices
                    
                if not training_mode and self._is_spec_decoding and self._is_last_block:
                    original_hidden_states = output_hidden_states
                    batch_size, seq_len, hidden_size = original_hidden_states.shape
                    device = original_hidden_states.device
                    valid_mask = keep_indices >= 0
                    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(keep_indices)
                    valid_hidden_states = original_hidden_states[batch_idx[valid_mask], keep_indices[valid_mask], :]
                    output_hidden_states = valid_hidden_states.unsqueeze(0)
                    
                self._last_keep_indices = keep_indices + cache_len
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

    def _normalize_keep_indices(
        self,
        keep_indices: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalize keep_indices to shape [B, L] int64 on target device.
        Falls back to identity indices when input is missing/invalid.
        """
        def _default_keep() -> torch.Tensor:
            return torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)

        if keep_indices is None or is_dummy(keep_indices):
            return _default_keep()

        if not torch.is_tensor(keep_indices):
            keep_indices = torch.as_tensor(keep_indices, device=device)
        else:
            keep_indices = keep_indices.to(device)

        if keep_indices.numel() == 0:
            return _default_keep()

        keep_indices = keep_indices.to(dtype=torch.int64)

        if keep_indices.ndim == 0:
            keep_indices = keep_indices.view(1, 1).expand(batch_size, 1)
        elif keep_indices.ndim == 1:
            keep_indices = keep_indices.view(1, -1).expand(batch_size, -1)
        elif keep_indices.ndim > 2:
            keep_indices = keep_indices.reshape(keep_indices.shape[0], -1)

        if keep_indices.shape[0] == 1 and batch_size > 1:
            keep_indices = keep_indices.expand(batch_size, -1)
        elif keep_indices.shape[0] != batch_size:
            logger.debug(
                "keep_indices batch mismatch in backend %s: got %s, expected %s; using default keep indices",
                self.name,
                tuple(keep_indices.shape),
                batch_size,
            )
            return _default_keep()

        return keep_indices

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
        device: torch.device,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(prefill_length):
            prefill_length = torch.as_tensor(prefill_length, device=device)
        else:
            prefill_length = prefill_length.to(device)

        if batch_size is None:
            if prefill_length.ndim >= 1 and prefill_length.shape[0] > 0:
                batch_size = int(prefill_length.shape[0])
            elif torch.is_tensor(kv_valid_lengths) and kv_valid_lengths.ndim >= 1 and kv_valid_lengths.shape[0] > 0:
                batch_size = int(kv_valid_lengths.shape[0])
            else:
                batch_size = 1

        # Normalize lengths to [B] to avoid shape mismatch in micro-batch/spec-dec mixed paths
        prefill_cap = int(torch.nan_to_num(prefill_length.float(), nan=0.0).max().item()) if prefill_length.numel() > 0 else 0
        prefill_cap = max(0, prefill_cap)
        prefill_length = self._normalize_kv_valid_lengths(
            kv_valid_lengths=prefill_length,
            batch_size=batch_size,
            max_kv_len=max(prefill_cap, 1_000_000),
            device=device,
        )

        if not torch.is_tensor(kv_valid_lengths):
            kv_valid_lengths = torch.as_tensor(kv_valid_lengths, device=device)
        else:
            kv_valid_lengths = kv_valid_lengths.to(device)

        kv_cap = int(torch.nan_to_num(kv_valid_lengths.float(), nan=0.0).max().item()) if kv_valid_lengths.numel() > 0 else 0
        kv_cap = max(prefill_cap, kv_cap, 0)
        kv_valid_lengths = self._normalize_kv_valid_lengths(
            kv_valid_lengths=kv_valid_lengths,
            batch_size=batch_size,
            max_kv_len=max(kv_cap, 1_000_000),
            device=device,
        )
        
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
        
    def _create_tree_position_ids_with_invalid_cache(
        self, 
        width: int, 
        depth: int, 
        prefill_length: torch.Tensor,          # (B,) 每个 batch 的实际 prompt 长度
        kv_cache_position_ids: Optional[torch.Tensor],  # (B, max_pos_len) 或 None, -1 是 padding
        batch_offset,
        device: torch.device,
        target_seq_len: Optional[int] = None,  # 目标序列长度（hidden states 的长度）
    ) -> torch.Tensor:
        B = prefill_length.shape[0]
        
        if isinstance(device, str):
            device = torch.device(device)
        
        # 1. 生成 Tree 模板
        tree_position_ids_list = []
        def dfs_generate(node_depth, current_depth):
            tree_position_ids_list.append(node_depth)
            if current_depth < depth:
                for _ in range(width):
                    dfs_generate(node_depth + 1, current_depth + 1)
        dfs_generate(0, 0)
        tree_len = len(tree_position_ids_list)
        tree_position_ids = torch.tensor(tree_position_ids_list, dtype=torch.long, device=device)
        
        # 2. 判断是否为 Prefill 阶段
        is_prefill = (kv_cache_position_ids is None or kv_cache_position_ids.numel() == 0)
        
        prefill_length = prefill_length.to(device)
        
        if is_prefill:
            # Prefill 阶段
            # 使用 target_seq_len 作为总长度（如果提供），否则使用 max_prefill_len + tree_len
            max_prefill_len = prefill_length.max().item()
            
            if target_seq_len is not None:
                total_len = target_seq_len
                # 从 target_seq_len 反推 prompt 部分的长度
                prompt_part_len = target_seq_len - tree_len
            else:
                total_len = max_prefill_len + tree_len
                prompt_part_len = max_prefill_len
            
            full_position_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
            
            # Prompt 部分的 position ids
            if prompt_part_len > 0:
                prefill_positions = torch.arange(prompt_part_len, dtype=torch.long, device=device)
                full_position_ids[:, :prompt_part_len] = prefill_positions.unsqueeze(0)
            
            # Tree 部分的 position ids
            # tree_base 是每个 batch 的 prompt 结束位置
            tree_base = prefill_length.unsqueeze(1)
            full_position_ids[:, prompt_part_len:] = tree_base + tree_position_ids.unsqueeze(0)
            
            return full_position_ids
        
        else:
            # Generation 阶段：基于有效 token 数量计算 position
            kv_cache_position_ids = kv_cache_position_ids.to(device)
            kv_cache_position_ids = self._slice_batch_aligned(kv_cache_position_ids, batch_offset, batch_offset + B, kv_cache_position_ids.shape[0])
            valid_mask = kv_cache_position_ids >= 0  # (B, max_pos_len)
            
            # 计算每个 batch 的有效 token 数量
            # 有效数量 = root_position + kv_cache_position_ids 中的有效值数量
            batch_indices = torch.arange(B, device=device)
            first_valid_idx = valid_mask.int().argmax(dim=1)
            root_positions = kv_cache_position_ids[batch_indices, first_valid_idx]  # (B,)
            
            tree_valid_counts = valid_mask.sum(dim=1)  # (B,)
            
            # 有效 token 总数 = root_position + tree_valid_counts
            effective_token_counts = root_positions + tree_valid_counts  # (B,)
            
            # 新 token 的 base position = 有效 token 总数
            base_positions = effective_token_counts  # (B,)
            
            # 生成 position_ids
            # 如果指定了 target_seq_len，使用它；否则使用 tree_len
            if target_seq_len is not None:
                actual_tree_len = target_seq_len
                # 如果 target_seq_len > tree_len，需要扩展 tree_position_ids
                if actual_tree_len > tree_len:
                    # 扩展时用最后一个值填充（或者其他逻辑）
                    extended_tree = torch.zeros(actual_tree_len, dtype=torch.long, device=device)
                    extended_tree[:tree_len] = tree_position_ids
                    # 填充剩余部分（保持最后的深度）
                    if tree_len > 0:
                        extended_tree[tree_len:] = tree_position_ids[-1]
                    tree_position_ids = extended_tree
                elif actual_tree_len < tree_len:
                    tree_position_ids = tree_position_ids[:actual_tree_len]
            
            position_ids = base_positions.unsqueeze(1) + tree_position_ids.unsqueeze(0)
            
            return position_ids
        
    def _slice_batch_aligned(
        self,
        value: Any,
        mb_start: int,
        mb_end: int,
        full_batch_size: int,
    ) -> Any:
        """
        Slice tensor-like request fields only if they are batch-aligned.
        Non-tensor / scalar / already-global fields are returned as-is.
        """
        if value is None or not torch.is_tensor(value):
            return value
        if is_dummy(value):
            return value
        if value.ndim == 0:
            return value
        if value.shape[0] == full_batch_size:
            return value[mb_start:mb_end].contiguous()
        return value

    def _normalize_kv_valid_lengths(
        self,
        kv_valid_lengths: torch.Tensor,
        batch_size: int,
        max_kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalize kv_valid_lengths to shape [B] (dtype=torch.long), clamped to [0, max_kv_len].
        This keeps attention mask logic robust across micro-batch/spec-dec mixed paths.
        """
        if not torch.is_tensor(kv_valid_lengths):
            kv_valid_lengths = torch.as_tensor(kv_valid_lengths, device=device)
        else:
            kv_valid_lengths = kv_valid_lengths.to(device)

        def _infer_from_vector(vec: torch.Tensor) -> int:
            if vec.numel() == 0:
                return 0
            if torch.is_floating_point(vec):
                inferred = int(torch.nan_to_num(vec, nan=0.0).max().item())
            else:
                nonneg = vec >= 0
                if nonneg.any():
                    masked = torch.where(nonneg, vec, torch.full_like(vec, -1))
                    max_based = int(masked.max().item()) + 1
                    count_based = int(nonneg.sum().item())
                    inferred = max(max_based, count_based)
                else:
                    inferred = 0
            return max(0, min(inferred, max_kv_len))

        if kv_valid_lengths.ndim == 0:
            kv_valid_lengths = kv_valid_lengths.view(1).expand(batch_size)

        elif kv_valid_lengths.ndim == 1:
            if kv_valid_lengths.numel() == 1:
                kv_valid_lengths = kv_valid_lengths.expand(batch_size)
            elif kv_valid_lengths.numel() != batch_size:
                inferred = _infer_from_vector(kv_valid_lengths)
                kv_valid_lengths = torch.full(
                    (batch_size,), inferred, dtype=torch.long, device=device
                )

        else:
            if kv_valid_lengths.shape[0] == batch_size:
                # Typical accidental shape: [B, L] (e.g. indices table).
                if kv_valid_lengths.shape[1] == 1:
                    kv_valid_lengths = kv_valid_lengths[:, 0]
                else:
                    if torch.is_floating_point(kv_valid_lengths):
                        kv_valid_lengths = torch.nan_to_num(kv_valid_lengths, nan=0.0).max(dim=1).values
                    else:
                        nonneg = kv_valid_lengths >= 0
                        masked = torch.where(nonneg, kv_valid_lengths, torch.full_like(kv_valid_lengths, -1))
                        max_based = masked.max(dim=1).values + 1
                        count_based = nonneg.sum(dim=1)
                        kv_valid_lengths = torch.maximum(max_based.to(torch.long), count_based.to(torch.long))
            else:
                flat = kv_valid_lengths.reshape(-1)
                if flat.numel() == batch_size:
                    kv_valid_lengths = flat
                elif flat.numel() == 1:
                    kv_valid_lengths = flat.expand(batch_size)
                else:
                    inferred = _infer_from_vector(flat)
                    kv_valid_lengths = torch.full(
                        (batch_size,), inferred, dtype=torch.long, device=device
                    )

        kv_valid_lengths = kv_valid_lengths.to(dtype=torch.long, device=device).contiguous()
        kv_valid_lengths = kv_valid_lengths.clamp(min=0, max=max_kv_len)
        if kv_valid_lengths.numel() != batch_size:
            fallback = int(kv_valid_lengths.max().item()) if kv_valid_lengths.numel() > 0 else 0
            fallback = max(0, min(fallback, max_kv_len))
            kv_valid_lengths = torch.full((batch_size,), fallback, dtype=torch.long, device=device)
        return kv_valid_lengths
        
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
