from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
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
    """Compute SHA256 hash of tensor for debugging - 优化CPU转换"""
    if tensor is None:
        return "None"
    try:
        # 优化：只在debug时才进行CPU转换，避免不必要的性能开销
        if not getattr(compute_tensor_hash, '_debug_enabled', False):
            return "hash_disabled"  # 在生产环境中禁用hash计算
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    except:
        return "error"
        
# 可以通过设置这个标志来启用/禁用hash计算
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
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
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
        # 🚀 Optimized: Add fast-path check to avoid repeated device comparison
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
            step_start_time = time.perf_counter()
            assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]" # Ensure hidden states are 3-dimensional
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            # Block-level debug output removed
            
            self._ensure_model_on_device()
            
            with self.cache_manager.use_cache(
                *inference_info.cache_handles  # Use cache to reduce memory requirements
            ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter): # Use adapter for inference


                # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
                # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
                # is at least 4-6x less than `autograd_memory`.
                max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info) # Estimate maximum chunk length
                # Debug output removed
                # see_memory_usage("transformer backend inference step : seq_len")
                output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None # Initialize output states
                # print("transformer backend inference step : output_hidden_states", output_hidden_states) # output_hidden_states:None
                # Centralized select: aggregate + reorder + slice
                selected = self.cache_manager.select_cache(
                    prefix_length=inference_info.prefix_length,
                    hypo_ids=hypo_ids,
                )
                layer_past = selected
                
                # 🔧 Add layer_past debug information
                # offload_logger.info(f"Select layer_past:")
                # offload_logger.info(f"   - layer_past type: {type(layer_past)}")
                # offload_logger.info(f"   - layer_past length: {len(layer_past) if layer_past else 0}")
                # if layer_past and len(layer_past) > 0:
                #     offload_logger.info(f"   - first tensor shape: {layer_past[0].shape}")
                #     offload_logger.info(f"   - first tensor device: {layer_past[0].device}")
                
                for offset in range(0, seq_len, max_chunk_length): # Iterate through sequence to process hidden states in chunks   only run offset=0
                    hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :] # Get current hidden states chunk
                    # print('transformer backend inference step() offset ', offset )
                    # print('transformer backend inference step() offset + max_chunk_length',  (offset + max_chunk_length))
                    
                    #  Generate correct position_ids for this chunk
                    chunk_length = min(max_chunk_length, seq_len - offset)
                    # 🚀 Optimized: Reuse cached position_ids base tensor
                    cache_key = (chunk_length, batch_size, hidden_states.device)
                    if cache_key not in self._position_ids_cache:
                        # Create base position_ids (0 to chunk_length-1)
                        base_ids = torch.arange(0, chunk_length, device=hidden_states.device, dtype=torch.long)
                        self._position_ids_cache[cache_key] = base_ids.unsqueeze(0).expand(batch_size, -1)
                    
                    # Add offset to cached base tensor (avoids creating new tensor)
                    position_ids = self._position_ids_cache[cache_key] + (inference_info.prefix_length + offset)
                    
                    # print(f' Generated position_ids for chunk: shape={position_ids.shape}, content={position_ids}')
                    
                    # 🔧 Add chunk processing debug information
                    # offload_logger.info(f" Processing chunk {offset//max_chunk_length + 1}:")
                    # offload_logger.info(f"   - chunk_length: {chunk_length}")
                    # offload_logger.info(f"   - hidden_states_chunk device: {hidden_states_chunk.device}")
                    # offload_logger.info(f"   - position_ids range: {position_ids.min().item()}-{position_ids.max().item()}")
                    
                    try:
                        # Fixed: Properly handle forward method return values with position_ids
                        # print(f' About to call module.forward with position_ids...')
                        forward_result = self.module.forward(
                            hidden_states_chunk, 
                            layer_past=layer_past, 
                            use_cache=True,  #  Keep use_cache=True to get cache tensors
                            position_ids=position_ids  #  Pass the generated position_ids
                        )
                        # print(f' module.forward returned: {type(forward_result)}, length: {len(forward_result) if forward_result else "None"}')
                        
                        if forward_result is None:
                            # print(f' ERROR: module.forward returned None!')
                            return (hidden_states,)  # Return original input as fallback
                        
                        output_hidden_states_chunk, new_kvs = forward_result
                        # print(f' Successfully unpacked: output_hidden_states_chunk={output_hidden_states_chunk.shape if output_hidden_states_chunk is not None else None}')
                        
                        # 🔧 Add forward result debug information
                        # offload_logger.info(f" module.forward completed:")
                        # offload_logger.info(f"   - output_hidden_states_chunk shape: {output_hidden_states_chunk.shape if output_hidden_states_chunk is not None else None}")
                        # offload_logger.info(f"   - new_kvs length: {len(new_kvs) if new_kvs else 0}")
                        # if new_kvs and len(new_kvs) > 0:
                        #     offload_logger.info(f"   - new_kvs[0] shape: {new_kvs[0].shape}")
                        #     offload_logger.info(f"   - new_kvs[0] device: {new_kvs[0].device}")
                        
                    except Exception as e:
                        # print(f' ERROR in module.forward: {type(e).__name__}: {e}')
                        # import traceback
                        # traceback.print_exc()
                        return (hidden_states,)  # Return original input as fallback
                    
                    if seq_len > max_chunk_length:
                        output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk # Store output
                    else:
                        output_hidden_states = output_hidden_states_chunk  # saves one memcopy # Copy memory only once
                    # layer_past = new_kvs # Update cache state

                # 🔧 Fixed: Restore cache update logic  
                past_key_values_length = 0
                if layer_past is not None and len(layer_past) > 0:
                    past_key_values_length = layer_past[0].shape[2]

                # logger.info(f"inference_step, output_hidden_states: {output_hidden_states}")
                # Centralized KV update via KVCacheManager (logs OFFLOAD: KV write ...)
                self.cache_manager.update_cache(new_kvs, past_key_values_length)
                
                # Block-level output debug removed
                
                return (output_hidden_states,) # Return output hidden states
                
        except Exception as e:
            # print(f' CRITICAL ERROR in inference_step: {type(e).__name__}: {e}')
            # import traceback
            # traceback.print_exc()
            return (hidden_states,)  # Return original input as fallback

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

    # Selection is centralized in KVCacheManager.select_cache()

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
    # Debug output removed
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
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        # import pdb; pdb.set_trace()
        return (hidden_states,)
