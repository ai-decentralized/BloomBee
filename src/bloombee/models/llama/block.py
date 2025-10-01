"""
LLaMA intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
import time
import threading
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRMSNorm,
    repeat_kv,
    rotate_half,
)
import numpy as np
from bloombee.utils.cuda_graphs import make_inference_graphed_callable
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import fix_recursive_import, TorchTensor, TorchDevice
from bloombee.flexgen_utils.utils import ValueHolder, array_1d, array_2d, array_3d
from bloombee.models.llama.flex_llama import FLEX_LlamaAttention, FLEX_LlamaMLP, LlamaDecoderLayer, DUMMY_WEIGHT, apply_rotary_pos_emb, FLEX_LlamaRMSNorm
from bloombee.flexgen_utils.llama_config import get_llama_config, download_llama_weights
from bloombee.flexgen_utils.task import Task
from transformers import AutoTokenizer
import os
from bloombee.utils.memory_usage import see_memory_usage, nvidia_smi_usage, log_mem

# Global tokenizer singleton - avoid creating duplicate tokenizers for each layer
_global_tokenizer = None
_tokenizer_lock = threading.Lock() if 'threading' in sys.modules else None

def get_global_tokenizer(model_name='llama-7b-hf'):
    """Get globally shared tokenizer, initialize only once"""
    global _global_tokenizer
    if _global_tokenizer is None:
        try:
            _global_tokenizer = AutoTokenizer.from_pretrained(
                f"huggyllama/{model_name}", 
                padding_side="left", 
                legacy=False
            )
            _global_tokenizer.pad_token = '[PAD]'
            print(f"[TOKENIZER_INIT] Global tokenizer initialized for {model_name}")
        except Exception as e:
            print(f"[TOKENIZER_INIT] Failed to initialize global tokenizer: {e}")
            _global_tokenizer = None
    return _global_tokenizer

fix_recursive_import()

from pynvml import *



class OptimizedLlamaAttention(FLEX_LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotary_graph = None
        self.temp_hidden_states = ValueHolder()

    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        hidden_states: torch.Tensor,
        cache_read_buf: ValueHolder,
        weight_read_buf: ValueHolder,
        cache_write_buf: ValueHolder,
        k: Optional[int] = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        generated_tokens_num=0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        assert not output_attentions

        # print('🔧 OptimizedLlamaAttention.forward(): received position_ids:', position_ids)
        # if position_ids is not None:
        #     print(f'🔧 position_ids shape: {position_ids.shape}, dtype: {position_ids.dtype}')
        #     print(f'🔧 position_ids content: {position_ids}')

        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.long
            ).unsqueeze(0) # pyright: ignore[reportAssignmentType]
            # print(f'🔧 Generated fallback position_ids: {position_ids}')

        # print('🔧 Final position_ids before processing:', position_ids)

        #   Optimized: Avoid .item() CPU-GPU sync by using direct indexing
        # Most common case: 2D tensor [batch_size, seq_len]
        if position_ids.dim() == 2:
            start_position = position_ids[0, 0]  # Keep as tensor, no .item() sync!
        elif position_ids.dim() == 1:
            start_position = position_ids[0]  # Keep as tensor
        elif position_ids.dim() == 0:
            start_position = position_ids  # Already scalar tensor
        elif position_ids.numel() == 0 or generated_tokens_num == 0:
            start_position = 0
        else:
            start_position = 0

        # print(f'🔧 Extracted start_position: {start_position}')

        self.temp_hidden_states.val = super(OptimizedLlamaAttention, self).forward(
            hidden_states, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, start_position, k
        )
        return self.temp_hidden_states.val, None, None


class OptimizedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_id: int, env: ExecutionEnv, policy: Policy, weight_home: array_1d, path: str):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.config = config
        self.env = env
        self.policy = policy

        self.self_attn = OptimizedLlamaAttention(config=config, env=env, policy=policy, layer_id=self.layer_id)
        self.mlp = FLEX_LlamaMLP(config=config, env=env, policy=policy, layer_id=self.layer_id)

        self.input_layernorm = FLEX_LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FLEX_LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_attn_graph = None
        self.post_attn_graph = None

        self.llama_config = config
        self.path = path
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(self.self_attn)
        layers.append(self.mlp)

        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None

        # Use globally shared tokenizer to avoid creating duplicate tokenizers for each layer
        self._cached_tokenizer = None
        # Don't initialize tokenizer in __init__ to avoid creating one for each layer
        
        # Improved Task management
        self._cached_task = None
        self._is_initialized = False
        self._test_inputs_cache = {}  # Optimization 3: cache test_inputs results
        
        # Cache frequently used calculation results
        self._last_prompt_len = None
        self._last_gen_len = None
        
        # Object reuse and caching strategy
        self._cached_torch_device = None
        self._cached_hidden_array = None
        self._last_gen_len_for_hidden = None
        self._cache_cleared = False
        
        # GPU stream management optimization
        self._streams_initialized = False

        # log_mem(f"[LlamaDecoderLayer:{self.layer_id}] before init_all_weights")
        self.init_all_weights()
        # log_mem(f"[LlamaDecoderLayer:{self.layer_id}] after init_all_weights")

        self.temp_hidden = ValueHolder()
        
        # Lazy initialization of GPU streams
        self._init_gpu_streams_if_needed()

    def _get_tokenizer(self):
        """Optimization: use globally shared tokenizer to avoid duplicate creation"""
        if self._cached_tokenizer is None:
            model_name = getattr(self.llama_config, 'name', 'llama-7b-hf')
            self._cached_tokenizer = get_global_tokenizer(model_name)
        return self._cached_tokenizer

    def _get_cached_test_inputs(self, prompt_len, num_prompts):
        """Optimization: cache test_inputs results to avoid duplicate calculations"""
        cache_key = (prompt_len, num_prompts)
        if cache_key not in self._test_inputs_cache:
            tokenizer = self._get_tokenizer()
            if tokenizer is not None:
                self._test_inputs_cache[cache_key] = get_test_inputs(
                    prompt_len, num_prompts, tokenizer
                )
            else:
                # If tokenizer is unavailable, use simple default values
                self._test_inputs_cache[cache_key] = ([0],) * num_prompts
        return self._test_inputs_cache[cache_key]

    def _should_rebuild_task(self, max_new_tokens, actual_prompt_len):
        """Optimization: simplify Task rebuild logic, reduce duplicate checks"""
        if self._cached_task is None:
            return True
        
        # Only rebuild when there are actual changes
        if (self._cached_task.gen_len != max_new_tokens or 
            self._cached_task.prompt_len != actual_prompt_len):
            return True
            
        return False
    
    def _should_force_cache_clear(self):
        """Determine if cache needs to be force cleared"""
        # Only clear cache when truly necessary
        return (self._cached_task is None or 
                self._last_prompt_len != self._cached_task.prompt_len or
                self._last_gen_len != self._cached_task.gen_len)

    def _init_gpu_streams_if_needed(self):
        """Lazy initialization of GPU streams to avoid duplicate creation"""
        if not self._streams_initialized:
            if not hasattr(self, 'load_weight_stream'):
                self.load_weight_stream = torch.cuda.Stream()
            if not hasattr(self, 'load_cache_stream'):
                self.load_cache_stream = torch.cuda.Stream()
            if not hasattr(self, 'store_cache_stream'):
                self.store_cache_stream = torch.cuda.Stream()
            self._streams_initialized = True

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def init_weight(self, j):
        model_name = os.path.basename(self.llama_config._name_or_path.rstrip('/'))
        self.llama_config.name = model_name
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{model_name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_llama_weights(self.llama_config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def _optimized_input_layernorm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.pre_attn_graph(hidden_states)

    def _optimized_output_layernorm(self, hidden_states):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.post_attn_graph(hidden_states)

    def update_attention_mask(self, gererated_tokens_num, k, mask_length):
        if gererated_tokens_num > 0:
            mask = self.attention_mask[k]
            if mask.val is not None:
                mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
                return

        gpu_batch_size = self.policy.gpu_batch_size

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
                             else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, mask_length), bool)
        mask_data = np.ones((gpu_batch_size, mask_length), dtype=bool)
        val.load_from_np(mask_data)
        # print(f"update_attention_mask, mask_length: {mask_length}, val: {val}")
        self.attention_mask[k].store(val)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        max_new_tokens: int = 1,
        do_sample: bool = True,
        temperature: float = 0.6,
        stop: Optional[int] = None,
        debug_mode: Optional[str] = None,
        cut_gen_len: Optional[int] = None,
        top_p: float = 0.9,
        verbose: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # log_mem(f"[Layer:{self.layer_id}] forward(start) batch={hidden_states.shape[0]} seq={hidden_states.shape[1]}")

        # Use globally shared tokenizer to avoid duplicate creation
        # Only get tokenizer when truly needed
        # if self._cached_tokenizer is None:
        #     self._cached_tokenizer = self._get_tokenizer()

        num_prompts = 1
        actual_prompt_len = hidden_states.shape[1] if hidden_states.shape[1] > 0 else 1
        prompt_len, gen_len, cut_gen_len = actual_prompt_len, max_new_tokens, max_new_tokens

        # Use simplified Task rebuild logic and add performance monitoring
        task_rebuild_start = None
        if self._should_rebuild_task(max_new_tokens, actual_prompt_len):
            task_rebuild_start = time.time()
            
            # Use cached test_inputs
            inputs = self._get_cached_test_inputs(prompt_len, num_prompts)

            self._cached_task = Task(
                inputs=inputs,
                prompt_len=len(inputs[0]),
                gen_len=max_new_tokens,
                cut_gen_len=cut_gen_len,
                do_sample=do_sample,
                temperature=temperature,
                stop=stop,
                top_p=top_p
            )
            
            # Cache parameters for next comparison
            self._last_prompt_len = actual_prompt_len
            self._last_gen_len = max_new_tokens
            
            if not self._is_initialized:
                self._is_initialized = True
                
            # Performance monitoring: record Task rebuild time
            if task_rebuild_start is not None:
                task_rebuild_time = (time.time() - task_rebuild_start) * 1000
                if task_rebuild_time > 1.0:  # 只记录超过1ms的情况
                    print(f"[BLOCK_PERF] Layer {self.layer_id} Task rebuild took: {task_rebuild_time:.3f}ms")

        task = self._cached_task

        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        num_prompts = len(task.inputs)
        prompt_len, gen_len = task.prompt_len, task.gen_len

        self.output_ids = np.ones((num_prompts, prompt_len + gen_len), dtype=np.int64)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)

        # Smart cache clearing - avoid clearing every time
        cache_clear_start = time.time()
        if not self._cache_cleared or self._should_force_cache_clear():
            num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.cache_read_buf[j][k].clear()
                    self.cache_write_buf[j][k].clear()
            for j in range(num_layers):
                self.weight_read_buf[j].clear()
            for k in range(num_gpu_batches):
                self.attention_mask[k].clear()
            self._cache_cleared = True
            
        cache_clear_time = (time.time() - cache_clear_start) * 1000
        if cache_clear_time > 2.0:
            print(f"[FLEXGEN_PERF] Layer {self.layer_id} Cache clear took: {cache_clear_time:.3f}ms")

        # Smart hidden array reuse
        hidden_alloc_start = time.time()
        if (self._cached_hidden_array is None or 
            self._last_gen_len_for_hidden != gen_len):
            self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)
            self._cached_hidden_array = self.hidden
            self._last_gen_len_for_hidden = gen_len
        else:
            self.hidden = self._cached_hidden_array
            
        hidden_alloc_time = (time.time() - hidden_alloc_start) * 1000
        if hidden_alloc_time > 2.0:
            print(f"[FLEXGEN_PERF] Layer {self.layer_id} Hidden array alloc took: {hidden_alloc_time:.3f}ms")

        # TorchDevice object reuse
        device_wrap_start = time.time()
        data = hidden_states
        if (self._cached_torch_device is None or 
            self._cached_torch_device.name != str(data.device)):
            self._cached_torch_device = TorchDevice(data.device)
        device = self._cached_torch_device
        
        tensor_data = TorchTensor(shape=data.shape, data=data, dtype=data.dtype, device=device)
        self.hidden[0][0][0].store(tensor_data)
        
        device_wrap_time = (time.time() - device_wrap_start) * 1000
        if device_wrap_time > 2.0:
            print(f"[FLEXGEN_PERF] Layer {self.layer_id} Device wrap took: {device_wrap_time:.3f}ms")

        # print(f"num_gpu_batches: {self.num_gpu_batches}")
        # print(f"input batch size: {hidden_states.shape[0]}")
        # print(f"gpu_batch_size: {self.policy.gpu_batch_size}")

        # CPU cache compute workspace initialization optimization
        cpu_workspace_start = time.time()
        self.task = task
        self.set_task(task)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        cpu_workspace_time = (time.time() - cpu_workspace_start) * 1000
        if cpu_workspace_time > 5.0:
            print(f"[FLEXGEN_PERF] Layer {self.layer_id} CPU workspace init took: {cpu_workspace_time:.3f}ms")

        debug_mode = kwargs.get('debug_mode', None)
        overlap = self.policy.overlap if hasattr(self.policy, 'overlap') else False

        if debug_mode is None:
            if not overlap:
                if position_ids is not None and position_ids.numel() > 0:
                    #   Optimized: Avoid .item() sync
                    current_position = position_ids.flatten()[0]
                    # print(f'🔧 Using actual position from position_ids: {current_position}')
                else:
                    current_position = 0
                    # print(f'🔧 No position_ids provided, using fallback position: {current_position}')

                i = current_position

                for k in range(self.num_gpu_batches):
                    if i == 0:
                        mask_length = hidden_states.shape[1]
                    else:
                        mask_length = i + 1
                    self.update_attention_mask(0, k, mask_length)

                # Weight loading performance monitoring and optimization
                weight_load_start = time.time()
                for j in range(self.num_layers):
                    for k in range(self.num_gpu_batches):
                        self.load_weight(i, j, k, overlap=False)
                weight_load_time = (time.time() - weight_load_start) * 1000
                if weight_load_time > 10.0:
                    print(f"[FLEXGEN_PERF] Layer {self.layer_id} Weight loading took: {weight_load_time:.3f}ms")

                final_outputs = []
                generated_tokens_num = 0
                if position_ids is not None and position_ids.numel() > 0:
                    #   Optimized: Avoid .item() sync - keep as tensor for faster ops
                    generated_tokens_num = position_ids.flatten()[-1] - self.task.prompt_len + 1
                for k in range(self.num_gpu_batches):
                    for j in range(self.num_layers):

                        # Load current layer cache
                        # self.load_cache(i, j, k, overlap=False)
                        # self.load_hidden(i, j, k)
                        if j == 0 and past_key_value is not None:

                            past_key, past_value = past_key_value
                            # Normalize past shapes into [B, H, S, D]
                            # logger.info(f"before format past_key: {past_key.shape}")
                            if past_key.dim() == 3:
                                # from backend packed: [B*H, D, S] or [B*H, S, D]
                                bh, x1, x2 = past_key.shape
                                b = hidden_states.shape[0]
                                h = bh // b
                                d = self.self_attn.head_dim
                                s = x2 if x1 == d else x1
                                if x1 == d and x2 == s:
                                    k_bhsd = past_key.permute(0, 2, 1)
                                else:
                                    k_bhsd = past_key
                                v_bhsd = past_value if past_value.shape[1] == s else past_value.permute(0, 2, 1)
                                past_key = k_bhsd.view(b, h, s, d)
                                past_value = v_bhsd.view(b, h, s, d)
                            # Transform to FlexGen expected (s, b*h, d)
                            b, h, s, d = past_key.shape
                            # logger.info(f"after format past_key: {past_key.shape}")
                            #   Optimized: Use reshape instead of permute+contiguous+view
                            # reshape() will avoid copy when possible
                            past_k_new = past_key.permute(2, 0, 1, 3).reshape(s, b * h, d)
                            past_v_new = past_value.permute(2, 0, 1, 3).reshape(s, b * h, d)
                            # logger.info(f"past_key: {past_k_new.shape}")
                            self.cache_read_buf[0][0].store((past_k_new, past_v_new))

                        # log_mem(f"[Layer:{self.layer_id}] before self_attn layer={j} i={i} k={k}")
                        layer_output = self.compute_layer(i, j, k, position_ids=position_ids, generated_tokens_num=generated_tokens_num)
                        # log_mem(f"[Layer:{self.layer_id}] after self_attn/MLP layer={j} i={i} k={k}")

                        if j == 0:
                            k_new, v_new = self.cache_write_buf[0][0].pop()

                            # Support compressed KV: decompress to torch.Tensor when needed
                            try:
                                from bloombee.flexgen_utils.pytorch_backend import DeviceType
                                def to_torch_tensor(x):
                                    # If FlexGen compressed tensor, decompress
                                    if hasattr(x, 'device') and (
                                        getattr(getattr(x, 'device', None), 'device_type', None) == DeviceType.COMPRESSED
                                        or (hasattr(x, 'data') and isinstance(getattr(x, 'data'), tuple) and len(getattr(x, 'data')) == 3)
                                    ):
                                        return x.device.decompress(x)
                                    # If FlexGen TorchTensor, return underlying torch tensor
                                    return getattr(x, 'data', x)
                                k_new_tensor = to_torch_tensor(k_new)
                                v_new_tensor = to_torch_tensor(v_new)
                            except Exception:
                                # Fallback to raw data if decompress pathway is unavailable
                                k_new_tensor = getattr(k_new, 'data', k_new)
                                v_new_tensor = getattr(v_new, 'data', v_new)
                            # Backend expects new_kvs shapes:
                            #   key:   (b*h, d, s)
                            #   value: (b*h, s, d)
                            key = k_new_tensor.permute(1, 2, 0)  # → (b*h, d, s)
                            value = v_new_tensor.permute(1, 0, 2)  # → (b*h, s, d)
                            # print(f"decoder, k_new shaped for backend: {key.shape}, v_new: {value.shape}")
                            past_key_value = (key, value)

                            self.cache_write_buf[0][0].store((k_new, v_new))

                    # print(f"forward, layer_output: {layer_output}")
                    #   Optimized: Avoid clone if not necessary
                    # Only clone if tensor has grad or is a view that might be modified
                    if layer_output.data.requires_grad or layer_output.data._base is not None:
                        final_outputs.append(layer_output.data.clone())
                    else:
                        # Safe to use directly - no grad, not a view
                        final_outputs.append(layer_output.data)

        # print(f"final_outputs: {len(final_outputs)}")
        if len(final_outputs) == 1:
            hidden_states = final_outputs[0]
        else:
            hidden_states = torch.cat(final_outputs, dim=0)
        # print(f"final hidden_states: {hidden_states}")

        outputs = (hidden_states, past_key_value)
        # log_mem(f"[Layer:{self.layer_id}] forward(end) out_shape={hidden_states.shape}")
        # Remove empty_cache call from each forward to reduce GPU overhead
        # torch.cuda.empty_cache()  # 这会导致性能问题
        return outputs

    def load_weight(self, i, j, k, overlap=True):
        # Fine-grained weight loading monitoring
        individual_weight_start = time.time()
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        individual_weight_time = (time.time() - individual_weight_start) * 1000
        if individual_weight_time > 5.0:
            print(f"[FLEXGEN_PERF] Layer {self.layer_id} Individual weight load [j={j}, k={k}] took: {individual_weight_time:.3f}ms")

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        # self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])
        pass

    def load_cache(self, i, j, k, overlap=True):
        if i == 0:
            return

                # Cache loading monitoring
        cache_load_start = time.time()
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        cache_load_time = (time.time() - cache_load_start) * 1000
        if cache_load_time > 3.0:
            print(f"[FLEXGEN_PERF] Layer {self.layer_id} Cache load [i={i}, j={j}, k={k}] took: {cache_load_time:.3f}ms")


    def store_cache(self, i, j, k, overlap=True):
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # print(f"store_cache in block")
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
            # Remove unnecessary synchronization to reduce GPU blocking
            # torch.cuda.synchronize()  # 这会造成性能瓶颈
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos - 1:pos])
        else:
            val = self.hidden[0][j - 1][k].pop().move(dst)

        self.hidden[0][j][k].store(val)

    def load_hidden_mlp(self, i, j, k):
        self.hidden[0][j][k].store(self.temp_hidden.val)

    def store_hidden(self, i, j, k):
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        if j == self.num_layers - 1:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[0][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos + 1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos + 1] = ids
        else:
            x = self.hidden[0][j][k]
            if x.val:
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k, position_ids=None, generated_tokens_num=0):
        if j == 1:
            self.hidden[0][j][k].val = self.temp_hidden.val

        # print(f'🔧 compute_layer: i={i}, j={j}, k={k}, received position_ids={position_ids}')

        self.layers[j].forward(hidden_states=self.hidden[0][j][k],
                               cache_read_buf=self.cache_read_buf[j][k],
                               weight_read_buf=self.weight_read_buf[j],
                               cache_write_buf=self.cache_write_buf[j][k],
                               k=k,
                               attention_mask=self.attention_mask[k],
                               position_ids=position_ids,
                               generated_tokens_num=generated_tokens_num)

        self.temp_hidden.val = self.layers[j].temp_hidden_states.val
        return self.layers[j].temp_hidden_states.val


class WrappedLlamaBlock(OptimizedLlamaDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #   Performance optimization: Pre-allocate attention_mask cache
        self._attention_mask_cache = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_llama(past_key_value, batch_size, past_key_values_length)

        # print(f'🔧 WrappedLlamaBlock.forward: received position_ids={position_ids}')
        if position_ids is not None:
            pass
            # print(f'🔧 WrappedLlamaBlock.forward: position_ids shape={position_ids.shape}, content={position_ids}')

        # print(f"WrappedLlamaBlock, hidden_states: {hidden_states}, seq_length: {seq_length}, past_key_value: {past_key_value}")
        #   Optimized: Reuse cached attention_mask
        if attention_mask is None:
            cache_key = (batch_size, seq_length, past_key_values_length, hidden_states.device, hidden_states.dtype)
            if cache_key not in self._attention_mask_cache:
                base_mask = torch.ones(
                    (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
                )
                self._attention_mask_cache[cache_key] = _prepare_4d_causal_attention_mask(
                    attention_mask=base_mask,
                    input_shape=(batch_size, seq_length),
                    inputs_embeds=hidden_states,
                    past_key_values_length=past_key_values_length,
                )
            attention_mask = self._attention_mask_cache[cache_key]
        else:
            # If attention_mask is provided, prepare it (don't cache custom masks)
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=hidden_states,
                past_key_values_length=past_key_values_length,
            )

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states, past_key_value = outputs
        # print('block.py WrappedLlamaBlock forward : outputs ', hidden_states)
        # print(f"WrappedLlamaBlock.forward, past_key_value: {past_key_value}")
        # print('use_cache', use_cache)

        return outputs

    def _reorder_cache_from_bloom_to_llama(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        # If already in [B, H, S, D], return as-is
        if key_states.dim() == 4 and value_states.dim() == 4:
            return key_states, value_states
        # Otherwise, expect Bloom-style packed heads: key [B*H, D, S] or [B*H, S, D], value [B*H, S, D] or [B*H, D, S]
        if key_states.dim() == 3:
            bh, d1, d2 = key_states.shape
            # Make key [B*H, S, D]
            if d2 == self.self_attn.head_dim and d1 == seq_length:
                # currently [B*H, S, D] — ok
                key_bhsd = key_states
            elif d1 == self.self_attn.head_dim and d2 == seq_length:
                # currently [B*H, D, S] — permute
                #   Optimized: contiguous() only if needed by subsequent ops
                key_bhsd = key_states.permute(0, 2, 1)
            else:
                # Fallback: assume second dim is sequence
                key_bhsd = key_states.permute(0, 2, 1)

            # Value to [B*H, S, D]
            if value_states.shape[1] == seq_length:
                val_bhsd = value_states
            else:
                #   Optimized: contiguous() only if needed
                val_bhsd = value_states.permute(0, 2, 1)

            # Reshape into [B, H, S, D]
            h = self.self_attn.num_key_value_heads
            d = self.self_attn.head_dim
            key_out = key_bhsd.view(batch_size, h, seq_length, d)
            val_out = val_bhsd.view(batch_size, h, seq_length, d)
            return (key_out, val_out)
        # Unexpected shapes; return as-is to avoid crashes
        return key_states, value_states

    def _reorder_cache_from_llama_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    """Simplify test_inputs generation to reduce tokenizer call overhead"""
    # Directly create simple input_ids to avoid tokenizer processing
    # Use pad_token_id as default value
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0
    
    # Create simple input_ids list with length 1 (minimum valid length)
    simple_input_ids = [pad_token_id]
    return (simple_input_ids,) * num_prompts