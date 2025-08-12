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

logger = get_logger(__name__)

# 创建专门的offloading调试logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)

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
        assert isinstance(self.module, TensorParallel)
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

        # Decide device placement policy for module based on offloading policy
        offload_policy = cache_manager.offloading_policy
        is_offloading_mode = (
            offload_policy.cache_gpu_percent < 100
            or offload_policy.cache_cpu_percent > 0
            or offload_policy.cache_disk_percent > 0
            or offload_policy.compress_cache
        )

        if is_offloading_mode:
            # Keep torch modules on CPU to prevent tensor_parallel from migrating shards to CUDA.
            cpu_devices = [torch.device('cpu')]
            self.module.devices = cpu_devices
            if hasattr(self.module, 'module_shards'):
                for shard in self.module.module_shards:
                    shard.to('cpu')
            if hasattr(self.module, 'output_device_index'):
                self.module.output_device_index = 0

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
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
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
            print("transformer backend inference step : seq_len", seq_len)
            print(f"🔧 Backend inference_step: batch_size={batch_size}, seq_len={seq_len}, prefix_length={inference_info.prefix_length}")
            
            # 🔧 添加offloading调试信息
            offload_logger.info(f"   - batch_size: {batch_size}")
            offload_logger.info(f"   - seq_len: {seq_len}")
            offload_logger.info(f"   - prefix_length: {inference_info.prefix_length}")
            offload_logger.info(f"   - cache_handles数量: {len(inference_info.cache_handles)}")
            offload_logger.info(f"   - 当前设备: {hidden_states.device}")
            
            # see_memory_usage("transformer backend inference step : seq_len")
            
            
            self._ensure_model_on_device()
            
            with self.cache_manager.use_cache(
                *inference_info.cache_handles  # Use cache to reduce memory requirements
            ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter): # Use adapter for inference


                # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
                # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
                # is at least 4-6x less than `autograd_memory`.
                max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info) # Estimate maximum chunk length
                print("transformer backend inference step() : max_chunk_length", max_chunk_length)
                # see_memory_usage("transformer backend inference step : seq_len")
                output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None # Initialize output states
                # print("transformer backend inference step : output_hidden_states", output_hidden_states) # output_hidden_states:None
                # Centralized select: aggregate + reorder + slice
                selected = self.cache_manager.select_cache(
                    prefix_length=inference_info.prefix_length,
                    hypo_ids=hypo_ids,
                )
                layer_past = selected
                
                # 🔧 添加layer_past调试信息
                offload_logger.info(f"选择layer_past:")
                offload_logger.info(f"   - layer_past类型: {type(layer_past)}")
                offload_logger.info(f"   - layer_past长度: {len(layer_past) if layer_past else 0}")
                if layer_past and len(layer_past) > 0:
                    offload_logger.info(f"   - 第一个tensor形状: {layer_past[0].shape}")
                    offload_logger.info(f"   - 第一个tensor设备: {layer_past[0].device}")
                
                for offset in range(0, seq_len, max_chunk_length): # Iterate through sequence to process hidden states in chunks   only run offset=0
                    hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :] # Get current hidden states chunk
                    print('transformer backend inference step() offset ', offset )
                    print('transformer backend inference step() offset + max_chunk_length',  (offset + max_chunk_length))
                    
                    #  Generate correct position_ids for this chunk
                    chunk_length = min(max_chunk_length, seq_len - offset)
                    # Create position_ids starting from prefix_length + offset
                    position_ids = torch.arange(
                        inference_info.prefix_length + offset,
                        inference_info.prefix_length + offset + chunk_length,
                        device=hidden_states.device,
                        dtype=torch.long
                    ).unsqueeze(0).expand(batch_size, -1)
                    
                    print(f' Generated position_ids for chunk: shape={position_ids.shape}, content={position_ids}')
                    
                    # 🔧 添加chunk处理调试信息
                    offload_logger.info(f" 处理chunk {offset//max_chunk_length + 1}:")
                    offload_logger.info(f"   - chunk_length: {chunk_length}")
                    offload_logger.info(f"   - hidden_states_chunk设备: {hidden_states_chunk.device}")
                    offload_logger.info(f"   - position_ids范围: {position_ids.min().item()}-{position_ids.max().item()}")
                    
                    try:
                        # Fixed: Properly handle forward method return values with position_ids
                        print(f' About to call module.forward with position_ids...')
                        forward_result = self.module.forward(
                            hidden_states_chunk, 
                            layer_past=layer_past, 
                            use_cache=True,  #  Keep use_cache=True to get cache tensors
                            position_ids=position_ids  #  Pass the generated position_ids
                        )
                        print(f' module.forward returned: {type(forward_result)}, length: {len(forward_result) if forward_result else "None"}')
                        
                        if forward_result is None:
                            print(f' ERROR: module.forward returned None!')
                            return (hidden_states,)  # Return original input as fallback
                        
                        output_hidden_states_chunk, new_kvs = forward_result
                        print(f' Successfully unpacked: output_hidden_states_chunk={output_hidden_states_chunk.shape if output_hidden_states_chunk is not None else None}')
                        
                        # 🔧 添加forward结果调试信息
                        offload_logger.info(f" module.forward完成:")
                        offload_logger.info(f"   - output_hidden_states_chunk形状: {output_hidden_states_chunk.shape if output_hidden_states_chunk is not None else None}")
                        offload_logger.info(f"   - new_kvs长度: {len(new_kvs) if new_kvs else 0}")
                        if new_kvs and len(new_kvs) > 0:
                            offload_logger.info(f"   - new_kvs[0]形状: {new_kvs[0].shape}")
                            offload_logger.info(f"   - new_kvs[0]设备: {new_kvs[0].device}")
                        
                    except Exception as e:
                        print(f' ERROR in module.forward: {type(e).__name__}: {e}')
                        import traceback
                        traceback.print_exc()
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

                # Centralized KV update via KVCacheManager (logs OFFLOAD: KV write ...)
                self.cache_manager.update_cache(new_kvs, past_key_values_length)

                print('backend.py output_hidden_states.shape ', output_hidden_states.shape)
                
                # 🔧 添加最终结果调试信息
                offload_logger.info(f" inference_step完成:")
                offload_logger.info(f"   - output_hidden_states形状: {output_hidden_states.shape}")
                offload_logger.info(f"   - output_hidden_states设备: {output_hidden_states.device}")
                offload_logger.info(f"   - 剩余内存: {self.cache_manager.bytes_left() / (1024*1024*1024):.2f} GB")
                
                return (output_hidden_states,) # Return output hidden states
                
        except Exception as e:
            print(f' CRITICAL ERROR in inference_step: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
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
    print('............... come into the merge_inference_pools_inplace() ' )
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
        print('............... come into the _MergedInferenceStep __call__' )
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            print('............... come into the _MergedInferenceStep __call__ inference_info.uid ', inference_info.uid)
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        # import pdb; pdb.set_trace()
        return (hidden_states,)
