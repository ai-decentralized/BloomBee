"""
LLaMA intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
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
from bloombee.utils.memory_usage import see_memory_usage, nvidia_smi_usage

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

        print('🔧 OptimizedLlamaAttention.forward(): received position_ids:', position_ids)
        if position_ids is not None:
            print(f'🔧 position_ids shape: {position_ids.shape}, dtype: {position_ids.dtype}')
            print(f'🔧 position_ids content: {position_ids}')

        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.long
            ).unsqueeze(0) # pyright: ignore[reportAssignmentType]
            print(f'🔧 Generated fallback position_ids: {position_ids}')

        print('🔧 Final position_ids before processing:', position_ids)

        if position_ids.numel() == 0 or generated_tokens_num == 0:
            start_position = 0
            print('🔧 position_ids is empty, using start_position=0')
        elif position_ids.dim() == 0:
            start_position = int(position_ids.item())
            print(f'🔧 position_ids is scalar: {start_position}')
        elif position_ids.dim() == 1:
            start_position = int(position_ids[0].item())
            print(f'🔧 position_ids is 1D, using first element: {start_position}')
        elif position_ids.dim() == 2:
            start_position = int(position_ids[0, 0].item())
            print(f'🔧 position_ids is 2D [{position_ids.shape[0]}, {position_ids.shape[1]}], using first element: {start_position}')
            if position_ids.shape[1] <= 10:
                print(f'🔧 Full position sequence: {position_ids[0].tolist()}')
        else:
            start_position = 0
            print(f'🔧 position_ids has unexpected dimensions {position_ids.dim()}, using fallback start_position=0')

        print(f'🔧 Extracted start_position: {start_position}')

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

        self._cached_tokenizer = None
        self._cached_task = None
        self._is_initialized = False

        self.cache_manager = None
        self._init_cache_manager()

        self.init_all_weights()

        self.temp_hidden = ValueHolder()

    def _init_cache_manager(self):
        from bloombee.server.memory_cache_manager import init_cache_manager_shared
        from bloombee.server.cache_coordinator import get_cache_interface, create_device_info_from_policy

        self.cache_interface = get_cache_interface()
        if self.cache_interface is not None:
            self.cache_interface.register_layer(self.layer_id, {
                'layer_type': 'llama_decoder',
                'policy': self.policy
            })

        self.cache_manager = init_cache_manager_shared(self.policy, self.layer_id)

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
        print(f"update_attention_mask, mask_length: {mask_length}, val: {val}")
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

        if self._cached_tokenizer is None:
            self._cached_tokenizer = AutoTokenizer.from_pretrained(f"huggyllama/{self.llama_config.name}", padding_side="left", legacy=False)
            self._cached_tokenizer.pad_token = '[PAD]'
        tokenizer = self._cached_tokenizer

        num_prompts = 1

        actual_prompt_len = hidden_states.shape[1] if hidden_states.shape[1] > 0 else 1
        prompt_len, gen_len, cut_gen_len = actual_prompt_len, max_new_tokens, max_new_tokens

        print(f"prompt_len: {prompt_len}")
        print(f"gen_len: {gen_len}")
        print(f"hidden_states: {hidden_states}")

        task_changed = (self._cached_task is None or
                        self._cached_task.gen_len != max_new_tokens or
                        self._cached_task.prompt_len != actual_prompt_len)

        if task_changed:
            inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
            if not self._is_initialized:
                print('inputs shape and content:', inputs)
                print('inputs[0] length:', len(inputs[0]) if inputs else 0)

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
            if not self._is_initialized:
                print(f'Task created - prompt_len: {self._cached_task.prompt_len}, gen_len: {self._cached_task.gen_len}')
                self._is_initialized = True

        task = self._cached_task

        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        num_prompts = len(task.inputs)
        prompt_len, gen_len = task.prompt_len, task.gen_len

        self.output_ids = np.ones((num_prompts, prompt_len + gen_len), dtype=np.int64)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)

        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()

        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        data = hidden_states
        device = TorchDevice(data.device)
        tensor_data = TorchTensor(shape=data.shape, data=data, dtype=data.dtype, device=device)
        self.hidden[0][0][0].store(tensor_data)

        print(f"num_gpu_batches: {self.num_gpu_batches}")
        print(f"input batch size: {hidden_states.shape[0]}")
        print(f"gpu_batch_size: {self.policy.gpu_batch_size}")

        self.task = task
        self.set_task(task)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        debug_mode = kwargs.get('debug_mode', None)
        overlap = self.policy.overlap if hasattr(self.policy, 'overlap') else False

        if debug_mode is None:
            if not overlap:
                if position_ids is not None and position_ids.numel() > 0:
                    current_position = position_ids.flatten()[0].item()
                    print(f'🔧 Using actual position from position_ids: {current_position}')
                else:
                    current_position = 0
                    print(f'🔧 No position_ids provided, using fallback position: {current_position}')

                i = current_position

                for k in range(self.num_gpu_batches):
                    if i == 0:
                        mask_length = hidden_states.shape[1]
                    else:
                        mask_length = i
                    self.update_attention_mask(0, k, mask_length)

                for j in range(self.num_layers):
                    for k in range(self.num_gpu_batches):
                        self.load_weight(i, j, k, overlap=False)

                final_outputs = []
                generated_tokens_num = 0
                if position_ids is not None and position_ids.numel() > 0:
                    generated_tokens_num = position_ids.flatten()[-1].item() - self.task.prompt_len + 1
                for k in range(self.num_gpu_batches):
                    for j in range(self.num_layers):

                        # 加载当前层的缓存
                        # self.load_cache(i, j, k, overlap=False)
                        # self.load_hidden(i, j, k)
                        if j == 0 and past_key_value is not None:

                            past_key, past_value = past_key_value
                            # Normalize past shapes into [B, H, S, D]
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

                            past_k_new = past_key.permute(2, 0, 1, 3).contiguous().view(s, b * h, d)
                            past_v_new = past_value.permute(2, 0, 1, 3).contiguous().view(s, b * h, d)
                            self.cache_read_buf[0][0].store((past_k_new, past_v_new))

                        layer_output = self.compute_layer(i, j, k, position_ids=position_ids, generated_tokens_num=generated_tokens_num)

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
                            print(f"decoder, k_new shaped for backend: {key.shape}, v_new: {value.shape}")
                            past_key_value = (key, value)

                            self.cache_write_buf[0][0].store((k_new, v_new))

                    print(f"forward, layer_output: {layer_output}")
                    final_outputs.append(layer_output.data.clone())

        print(f"final_outputs: {len(final_outputs)}")
        if len(final_outputs) == 1:
            hidden_states = final_outputs[0]
        else:
            hidden_states = torch.cat(final_outputs, dim=0)
        print(f"final hidden_states: {hidden_states}")

        outputs = (hidden_states, past_key_value)
        torch.cuda.empty_cache()
        return outputs

    def load_weight(self, i, j, k, overlap=True):
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        if i == 0:
            return

        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

        if j == 0:
            cache_buf = self.cache_read_buf[j][k]
            if cache_buf.val is not None:
                k_cache, v_cache = cache_buf.val
                if i == 12 or i == 13:
                    print(f"after load cache, k_cache: {k_cache}, v_cache: {v_cache}")
                if isinstance(k_cache, tuple) and len(k_cache) == 2:
                    k_data = k_cache[0]
                    print(f"load_cache i={i}, j={j}, k={k}: k_cache.shape={k_data.shape if hasattr(k_data, 'shape') else 'no shape'}")
                    if hasattr(k_data, 'data'):
                        print(f"  k_cache stats: mean={k_data.data.mean():.6f}, std={k_data.data.std():.6f}")
                        print(f"  k_cache first few: {k_data.data.flatten()[:10]}")
                elif hasattr(k_cache, 'shape'):
                    print(f"load_cache i={i}, j={j}, k={k}: k_cache.shape={k_cache.shape}")
                    if hasattr(k_cache, 'data'):
                        print(f"  k_cache stats: mean={k_cache.data.mean():.6f}, std={k_cache.data.std():.6f}")
                        print(f"  k_cache first few: {k_cache.data.flatten()[:10]}")
                else:
                    print(f"load_cache i={i}, j={j}, k={k}: k_cache type={type(k_cache)}")
            else:
                print(f"load_cache i={i}, j={j}, k={k}: cache_read_buf.val is None")

    def store_cache(self, i, j, k, overlap=True):
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        print(f"store_cache in block")
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
            torch.cuda.synchronize()
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

        print(f'🔧 compute_layer: i={i}, j={j}, k={k}, received position_ids={position_ids}')

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

        print(f'🔧 WrappedLlamaBlock.forward: received position_ids={position_ids}')
        if position_ids is not None:
            print(f'🔧 WrappedLlamaBlock.forward: position_ids shape={position_ids.shape}, content={position_ids}')

        print(f"WrappedLlamaBlock, hidden_states: {hidden_states}, seq_length: {seq_length}, past_key_value: {past_key_value}")
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

        import logging
        offload_logger = logging.getLogger('bloombee.offloading')

        if position_ids is not None:
            if position_ids.shape == (1, 1) and position_ids[0][0].item() == 0:
                current_position = 0
            else:
                current_position = position_ids[0][0].item()
        else:
            if past_key_value is not None:
                current_position = past_key_value[0].shape[2]
            else:
                current_position = 0

        layer_id = getattr(self, 'layer_id', 0)

        offload_logger.info(f" position info - current_position:{current_position}, layer_id:{layer_id}")
        if position_ids is not None:
            offload_logger.info(f"   - position_ids shape: {position_ids.shape}")
            offload_logger.info(f"   - position_ids: {position_ids}")
        if past_key_value is not None:
            offload_logger.info(f"   - past_key_value length: {past_key_value[0].shape[2]}")

        if hasattr(self, 'cache_interface') and self.cache_interface is not None and current_position > 0:
            offload_logger.info(f" WrappedLlamaBlock.load_cache - pos:{current_position}, layer:{layer_id}")
            try:
                result = self.cache_interface.load_cache(layer_id, current_position, str(hidden_states.device), 0)
                if result is not None:
                    offload_logger.info(f" load cache - pos:{current_position}, layer:{layer_id}")
                else:
                    offload_logger.warning(f" cache not exist - pos:{current_position}, 层:{layer_id}")
            except Exception as e:
                offload_logger.warning(f"{e}")
        elif hasattr(self, 'cache_manager') and self.cache_manager is not None and current_position > 0:
            offload_logger.info(f" WrappedLlamaBlock.load_cache (fallback) - :{current_position}, :{layer_id}")
            try:
                result = self.cache_manager.load_cache(current_position, layer_id, 0)
                if result is not None:
                    offload_logger.info(f" 成功加载缓存 (fallback) - :{current_position}, :{layer_id}")
                else:
                    offload_logger.warning(f":{current_position}, :{layer_id}")
            except Exception as e:
                offload_logger.warning(f" KVCacheManager load_cache fail: {e}")

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
        print('block.py WrappedLlamaBlock forward : outputs ', hidden_states)
        print(f"WrappedLlamaBlock.forward, past_key_value: {past_key_value}")
        print('use_cache', use_cache)

        if hasattr(self, 'cache_interface') and self.cache_interface is not None:
            offload_logger.info(f"WrappedLlamaBlock.store_cache - pos:{current_position}, layer:{layer_id}")
            try:
                if past_key_value is not None:
                    handle = self.cache_interface.store_cache(layer_id, current_position, past_key_value, hidden_states.device, 0)
                    if handle is not None:
                        offload_logger.info(f"load cache - pos:{current_position}, layer:{layer_id}, handle:{handle}")
                    else:
                        offload_logger.warning(f"load cache fail:{current_position}, layer:{layer_id}")
                else:
                    offload_logger.warning(f" past_key_value is empty，skip store_cache")
            except Exception as e:
                offload_logger.warning(f"store_cache fail: {e}")
        elif hasattr(self, 'cache_manager') and self.cache_manager is not None:
            offload_logger.info(f"WrappedLlamaBlock.store_cache (fallback) - pos:{current_position}, layer:{layer_id}")
            try:
                from bloombee.server.memory_cache import UnifiedCache, DeviceInfo
                from bloombee.server.cache_coordinator import create_device_info_from_policy

                if past_key_value is not None:
                    device_info = create_device_info_from_policy(
                        self.policy if hasattr(self, 'policy') else None,
                        str(hidden_states.device)
                    )

                    unified_cache = UnifiedCache(
                        past_key_value=past_key_value,
                        device_info=device_info
                    )

                    self.cache_manager.store_cache(unified_cache, current_position, layer_id, 0)
                    offload_logger.info(f":{current_position}, :{layer_id}, :{device_info.device_type} ({device_info.device_id})")
                else:
                    offload_logger.warning(f"⚠️ past_key_value为空，跳过store_cache")
            except Exception as e:
                offload_logger.warning(f"⚠️ KVCacheManager store_cache失败: {e}")

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
                key_bhsd = key_states.permute(0, 2, 1).contiguous()
            else:
                # Fallback: assume second dim is sequence
                key_bhsd = key_states.permute(0, 2, 1).contiguous()

            # Value to [B*H, S, D]
            if value_states.shape[1] == seq_length:
                val_bhsd = value_states
            else:
                val_bhsd = value_states.permute(0, 2, 1).contiguous()

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
    prompts = [""]
    input_ids = tokenizer(prompts, padding=False, truncation=True).input_ids
    return (input_ids[0],) * num_prompts