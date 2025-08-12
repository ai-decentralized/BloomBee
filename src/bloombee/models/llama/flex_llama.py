"""
Usage:
python3 -m petals.models.llama.flex_llama --model huggingface repo --gpu-batch-size 32 --percent 100 0 100 0 100 0
modified on flex_opt.py
"""

import argparse
import dataclasses
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.llama_config import LlamaConfig, get_llama_config, download_llama_weights
from bloombee.flexgen_utils.pytorch_backend import fix_recursive_import, general_copy, DeviceType, TorchDevice, TorchTensor, TorchDisk, \
    TorchMixedDevice
from bloombee.flexgen_utils.utils import (GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)
from bloombee.flexgen_utils.task import Task
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from torch import nn
from transformers import AutoTokenizer
from bloombee.flexgen_utils.timer import timers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from bloombee.utils.memory_usage import see_memory_usage

import logging

# 创建专门的offloading调试logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)

fix_recursive_import()

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    # LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    repeat_kv,
    rotate_half,
)




DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

from pynvml import *

class FLEX_LlamaRMSNorm(LlamaRMSNorm): #put in fex_llama
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps=1e-6)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
    
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]

def init_weight_list(weight_specs, policy, env):
    
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    print('dev_percents :[ disk, cpu, gpu]', dev_percents)
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        # print('mid_percent ', mid_percent)
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]
        # print('weight_specs[i] ', weight_specs[i][2])
        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)
            # print('weight.shape', weight.shape)
            # print('weight', weight)

            if DUMMY_WEIGHT not in filename:
                try:
                    weight.load_from_np_file(weight_specs[i][2])
                except (FileNotFoundError, AttributeError) as e:
                    print(f"Warning: Could not load weight from file {weight_specs[i][2]}: {e}")
                    # 如果文件不存在或加载失败，使用随机初始化
                    weight.load_from_np(np.random.rand(*shape).astype(dtype))
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            # 检查压缩设备是否可用
            if hasattr(home, 'compressed_device') and home.compressed_device is not None:
                weight = home.compressed_device.allocate(
                    shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

                if DUMMY_WEIGHT not in filename:
                    try:
                        weight.load_from_np_file(weight_specs[i][2])
                    except (FileNotFoundError, AttributeError) as e:
                        print(f"Warning: Could not load weight from file {weight_specs[i][2]}: {e}")
                        # 如果文件不存在或加载失败，使用随机初始化
                        for i in range(2):
                            x = weight.data[i]
                            x.load_from_np(np.random.rand(*x.shape).astype(torch_dtype_to_np_dtype[x.dtype]))
                else:
                    for i in range(2):
                        x = weight.data[i]
                        x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
            else:
                # 如果压缩设备不可用，回退到非压缩方式
                print(f"Warning: Compressed device not available, falling back to non-compressed allocation")
                weight = home.allocate(shape, dtype, pin_memory=pin_memory)
                if DUMMY_WEIGHT not in filename:
                    try:
                        weight.load_from_np_file(weight_specs[i][2])
                    except (FileNotFoundError, AttributeError) as e:
                        print(f"Warning: Could not load weight from file {weight_specs[i][2]}: {e}")
                        # 如果文件不存在或加载失败，使用随机初始化
                        weight.load_from_np(np.random.rand(*shape).astype(dtype))
                else:
                    weight.load_from_np(np.ones(shape, dtype))
        # print('weight.data ', weight.data)
        ret.append(weight)
        
    return ret

# 添加一个新函数，用于从 PyTorch 模型加载权重到 FlexGen 格式
def load_weights_from_pytorch_model(model, policy, env, weight_home, block_index):
    """
    从 PyTorch 模型加载权重到 FlexGen 格式
    
    Args:
        model: PyTorch 模型
        policy: FlexGen 策略
        env: FlexGen 环境
        weight_home: 权重存储位置
        block_index: 块索引
    """
    weight_specs = []
    
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        # 创建权重规格
        shape = param.shape
        dtype = param.dtype
        # 使用参数名称作为文件名，确保唯一性
        filename = f"block_{block_index}_{name}"
        
        weight_specs.append((shape, dtype, filename))
        
        # 将参数移动到 CPU，避免在 GPU 上存储
        param.data = param.data.to('cpu')
    
    try:
        # 初始化权重列表
        weights = init_weight_list(weight_specs, policy, env)
        
        # 将权重加载到模型中
        for (name, _), weight in zip(model.named_parameters(), weights):
            param = getattr(model, name)
            param.data = weight.data.to(param.device)
        
        # 存储权重规格，以便后续使用
        weight_home[block_index] = weight_specs
        
        return weights
    except Exception as e:
        print(f"Warning: Failed to initialize weights with FlexGen: {e}")
        print("Falling back to direct parameter assignment")
        
        # 如果 FlexGen 初始化失败，直接使用参数赋值
        for name, param in model.named_parameters():
            # 确保参数在 CPU 上
            param.data = param.data.to('cpu')
        
        # 存储空的权重规格
        weight_home[block_index] = []
        
        return []

# class InputEmbed:
#     def __init__(self, config, env, policy):
#         self.config = config
#         self.env = env
#         self.policy = policy
#         self.compute = self.env.gpu
#         self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
#             else self.compute)

#         self.task = None
#         self.token_type_embeddings = nn.Embedding(
#             config.type_vocab_size, config.hidden_size, device="cuda:0"
#         )

#     def set_task(self, task):
#         self.task = task

#     def init_weight(self, weight_home, path):
#         v, h, dtype = (self.config.vocab_size, self.config.hidden_size, self.config.dtype)
#         path = os.path.join(path, "")
#         weight_specs = [
#             ((v, h), dtype, path + "embed_tokens.weight"),
#             ]
#         weights = init_weight_list(weight_specs, self.policy, self.env)
#         weight_home.store(weights)

#     def load_weight(self, weight_home, weight_read_buf, k):
#         w_token = weight_home.val[0]
#         if k == 0:
#             dst = self.weight_load_dst
#             weight_read_buf.store((w_token.smart_copy(dst)))

#     def init_cache_one_gpu_batch(self, cache_home):
#         pass  # do nothing

#     def load_cache(self, cache_home, cache_read_buf, i):
#         pass  # do nothing

#     def store_cache(self, cache_home, cache_write_buf, i):
#         pass  # do nothing

#     def input_act_shape_and_dtype(self, batch_size, seq_len):
#         return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

#     def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
#                 cache_write_buf, i, k):
#         # Compute input embedding 
#         donate = [False] * 3
#         h, donate[0] = hidden.val, True
#         mask, donate[1] = attention_mask.val.smart_copy(self.compute)

#         if k == self.policy.num_gpu_batches - 1:
#             # Clear the weight_read_buf if it is the last gpu batch
#             (w_token, donate[2]) = weight_read_buf.pop()
#         else:
#             (w_token, _) = weight_read_buf.val
#         h = self.compute.llama_input_embed(h, mask,
#             w_token, self.config.pad_token_id, donate, self.token_type_embeddings)
#         hidden.val = h


# class LlamaRMSNorm(nn.Module):
#     def __init__(
#         self,
#         config: LlamaConfig,
#         env: ExecutionEnv,
#         policy: Policy,
#         layer_id: int,
#     ):
#         super().__init__()
#         self.config = config
#         self.env = env
#         self.layer_id = layer_id
#         self.policy = policy
#         self.compute = self.env.gpu
#         self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
#                                 else self.compute)

#         self.task = None
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def set_task(self, task):
#         self.task = task

#     def init_weight(self, weight_home, path):
#         intermediate_size, h, dtype = (self.config.intermediate_size, self.config.hidden_size, self.config.dtype)
#         path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
#         weight_specs = [
#             # 4 weight files
#             # gate_proj
#             ((intermediate_size, h), dtype, path + "mlp.gate_proj.weight"),
#             # down_proj
#             ((h, intermediate_size), dtype, path + "mlp.down_proj.weight"),
#             # up_proj
#             ((intermediate_size, h), dtype, path + "mlp.up_proj.weight"),
#             # post attention layer norm
#             ((h, ), dtype, path + "post_attention_layernorm.weight"),
#         ]
#         weights = init_weight_list(weight_specs, self.policy, self.env)
#         weight_home.store(weights)

#     def load_weight(self, weight_home, weight_read_buf, k):
#         gate, down, up, post_attention_layernorm = weight_home.val
#         if k == 0:
#             dst1 = self.weight_load_dst
#             dst2 = self.compute
#             weight_read_buf.store((
#                     gate.smart_copy(dst1),
#                     down.smart_copy(dst1),
#                     up.smart_copy(dst1),
#                     post_attention_layernorm.smart_copy(dst2)
#             ))

#     def init_cache_one_gpu_batch(self, cache_home):
#         pass  # do nothing

#     def load_cache(self, cache_home, cache_read_buf, i):
#         pass  # do nothing

#     def store_cache(self, cache_home, cache_write_buf, i):
#         pass  # do nothing

#     def input_act_shape_and_dtype(self, batch_size, seq_len):
#         return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

#     def forward(self, 
#         x,
#         cache_read_buf,
#         weight_read_buf,
#         attention_mask,
#         cache_write_buf,
#         i=0,
#         k: int = 0
#         ):
#         donate = [False] * 9
#         h, donate[0] = x.val, True

#         if k == self.policy.num_gpu_batches - 1:
#             # Clear the weight_read_buf if it is the last gpu batch
#             ((gate, donate[1]), (down, donate[3]),
#              (up, donate[5]), (post_attention_layernorm, donate[7])) = weight_read_buf.pop()
#         else:
#             ((gate, _), (down, _),
#              (up, _), (post_attention_layernorm, _)) = weight_read_buf.val

#         h = self.compute.mlp_llama(h, gate, down, up, donate, self.config, post_attention_layernorm)
#         x.val = h



class FLEX_LlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        
        super().__init__(config)
        self.config = config
        self.llama_config = get_llama_config('huggyllama/llama-7b')
        self.num_heads = config.num_attention_heads
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
                                else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
                                  else self.env.gpu)
        
        self.task = None
        
        # 新增：支持KVCacheManager
        self.cache_manager = None
        self._init_cache_manager()

    def _init_cache_manager(self):
        """Initialize cache manager using shared utility"""
        from bloombee.server.memory_cache_manager import init_cache_manager_shared
        from bloombee.server.cache_coordinator import get_cache_interface, create_device_info_from_policy
        
        # 使用缓存协调器而不是直接持有cache_manager
        self.cache_interface = get_cache_interface()
        if self.cache_interface is not None:
            # 注册当前层到协调器
            self.cache_interface.register_layer(self.layer_id, {
                'layer_type': 'llama_attention',
                'policy': self.policy
            })
        
        # 保留原有的cache_manager用于向后兼容
        self.cache_manager = init_cache_manager_shared(self.policy, self.layer_id)

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, np.float16)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # 5 weight files
            # w_q
            ((h, h), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((h, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((h, h), dtype, path + "self_attn.v_proj.weight"),
            # w_out
            ((h, h), dtype, path + "self_attn.o_proj.weight"),
            # input layer norm
            ((h, ), dtype, path + "input_layernorm.weight"),
            # rotary_embed
            ((64, ), dtype, path + "self_attn.rotary_emb.inv_freq"),
        ]
        # see_memory_usage("-----------------------------------------before init weights of LLamaAttention ")
        weights = init_weight_list(weight_specs, self.policy, self.env)
        # see_memory_usage("-----------------------------------------after init weights of LLamaAttention ")
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, w_k, w_v, w_out, input_layernorm, rotary_emb_inv_freq = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_out.smart_copy(dst1),
                input_layernorm.smart_copy(dst2),
                rotary_emb_inv_freq.smart_copy(dst2),
            ))
            
    def init_cache_one_gpu_batch(self, cache_home):
        """
        初始化一个GPU批次的缓存
        支持KVCacheManager的统一接口
        """
        if self.cache_manager is not None:
            # 使用KVCacheManager的统一接口
            try:
                from bloombee.server.memory_cache import UnifiedCache
                
                unified_cache = self.cache_manager.init_cache_one_gpu_batch(
                    layer_id=self.layer_id,
                    batch_id=0,  # 这里需要根据实际情况调整batch_id
                    config=self.config,
                    task=self.task,
                    policy=self.policy
                )
                
                # 将UnifiedCache存储到cache_home
                cache_home.store(unified_cache)
                offload_logger.info(f"初始化缓存 - 层:{self.layer_id}")
                offload_logger.info(f"   - 使用KVCacheManager")
                return
                
            except Exception as e:
                offload_logger.warning(f"⚠️ KVCacheManager init_cache失败，使用原有实现: {e}")
        
        # 原有的实现作为fallback
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
            
        task_copy = Task(
                inputs=self.task.inputs,
                prompt_len=self.task.prompt_len,
                gen_len=max(self.task.gen_len, 128),
                cut_gen_len=self.task.cut_gen_len,
                do_sample=self.task.do_sample,
                temperature=self.task.temperature,
                stop=self.task.stop,
                top_p=self.task.top_p,
        )
        
        cache = device.init_cache_one_gpu_batch(self.config, task_copy, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        """
        加载缓存
        支持缓存协调器的统一接口
        """
        if i == 0:  # prefill, no cache
            return

        # 使用缓存协调器加载缓存
        if hasattr(self, 'cache_interface') and self.cache_interface is not None:
            try:
                # 确定目标设备
                target_device = 'cuda:0' if not self.policy.cpu_cache_compute else 'cpu'
                
                unified_cache = self.cache_interface.load_cache(
                    layer_id=self.layer_id,
                    position=i,
                    target_device=target_device,
                    batch_id=0
                )
                
                if unified_cache and unified_cache.past_key_value:
                    # 将UnifiedCache转换为cache_read_buf格式
                    cache_read_buf.store(unified_cache.past_key_value)
                    offload_logger.info(f"加载缓存 - 层:{self.layer_id}, 位置:{i}")
                    offload_logger.info(f"   - 使用缓存协调器成功")
                else:
                    # 处理缓存不存在的情况
                    cache_read_buf.store(None)
                    offload_logger.info(f" 加载缓存 - 层:{self.layer_id}, 位置:{i}")
                    offload_logger.info(f"   - 缓存不存在，使用缓存协调器")
                return
                
            except Exception as e:
                offload_logger.warning(f"⚠️ 缓存协调器load_cache失败，使用原有实现: {e}")
        
        # 向后兼容：使用原有的cache_manager
        if self.cache_manager is not None:
            try:
                from bloombee.server.memory_cache import UnifiedCache
                
                # 确定目标设备
                target_device = 'cuda:0' if not self.policy.cpu_cache_compute else 'cpu'
                
                unified_cache = self.cache_manager.load_cache(
                    position=i,
                    layer_id=self.layer_id,
                    batch_id=0,  # 需要根据实际情况调整
                    target_device=target_device
                )
                
                if unified_cache and unified_cache.past_key_value:
                    # 将UnifiedCache转换为cache_read_buf格式
                    cache_read_buf.store(unified_cache.past_key_value)
                    offload_logger.info(f"加载缓存 (fallback) - 层:{self.layer_id}, 位置:{i}")
                    offload_logger.info(f"   - 使用KVCacheManager成功")
                else:
                    # 处理缓存不存在的情况
                    cache_read_buf.store(None)
                    offload_logger.info(f" 加载缓存 (fallback) - 层:{self.layer_id}, 位置:{i}")
                    offload_logger.info(f"   - 缓存不存在，使用KVCacheManager")
                return
                
            except Exception as e:
                offload_logger.warning(f"⚠️ KVCacheManager load_cache失败，使用原有实现: {e}")

        # 原有的实现作为fallback
        k_home, v_home = cache_home.val
        
        cache_nan = torch.isnan(k_home.data).any()
        cache_inf = torch.isinf(k_home.data).any()
        print(f"load_cache[{i}]: cache_home NaN={cache_nan}, Inf={cache_inf}")
        
        if cache_nan:
            # 找出哪些位置有NaN
            nan_mask = torch.isnan(k_home.data)
            nan_positions = nan_mask.nonzero()[:5]  # 前5个NaN位置
            print(f"NaN positions in cache_home: {nan_positions}")
        
        # print(f"k_home: {k_home.data}, v_home: {v_home.data}")
        

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                        k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute
            
        print(f"load_cache path: {path}, current KV cache position: {i}, k_home.shape[1]: {k_home.shape[1]}")

        if path == 0:  # Direct copy
            # shape: (s, b * num_attention_heads, head_dim)
            indices = (slice(0, i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
            print(f"cache_read_buf: {cache_read_buf.val[0]}")
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * num_attention_heads, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * num_attention_heads, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        """
        存储缓存
        支持缓存协调器的统一接口
        """
        # 使用缓存协调器存储缓存
        if hasattr(self, 'cache_interface') and self.cache_interface is not None:
            try:
                # 从cache_write_buf获取新的缓存数据
                new_cache_data = cache_write_buf.pop()
                
                if new_cache_data is not None:
                    # 通过协调器存储缓存
                    handle = self.cache_interface.store_cache(
                        layer_id=self.layer_id,
                        position=i,
                        past_key_value=new_cache_data,
                        device=torch.device('cuda:0'),
                        batch_id=0
                    )
                    
                    if handle is not None:
                        offload_logger.info(f" 存储缓存 - 层:{self.layer_id}, 位置:{i}, 句柄:{handle}")
                        offload_logger.info(f"   - 使用缓存协调器成功")
                    else:
                        offload_logger.warning(f"⚠️ 存储缓存失败 - 层:{self.layer_id}, 位置:{i}")
                    return
                else:
                    offload_logger.warning(f"⚠️ new_cache_data为空，跳过存储")
                    return
                    
            except Exception as e:
                offload_logger.warning(f"⚠️ 缓存协调器store_cache失败，使用原有实现: {e}")
        
        # 向后兼容：使用原有的cache_manager
        if self.cache_manager is not None:
            try:
                from bloombee.server.memory_cache import UnifiedCache, DeviceInfo
                
                # 从cache_write_buf获取新的缓存数据
                new_cache_data = cache_write_buf.pop()
                
                if new_cache_data is not None:
                    # 创建UnifiedCache
                    # 使用统一的设备分配工具函数
                    device_info = create_device_info_from_policy(
                        self.policy if hasattr(self, 'policy') else None,
                        'cuda:0',
                        self.policy.comp_cache_config if hasattr(self, 'policy') and self.policy.compress_cache else None
                    )
                    
                    unified_cache = UnifiedCache(
                        past_key_value=new_cache_data,
                        device_info=device_info
                    )
                    
                    # 使用KVCacheManager存储
                    self.cache_manager.store_cache(
                        unified_cache=unified_cache,
                        position=i,
                        layer_id=self.layer_id,
                        batch_id=0  # 需要根据实际情况调整
                    )
                    offload_logger.info(f" 存储缓存 (fallback) - 层:{self.layer_id}, 位置:{i}, 设备:{device_info.device_type} ({device_info.device_id})")
                    offload_logger.info(f"   - 使用KVCacheManager成功")
                    return
                    
            except Exception as e:
                offload_logger.warning(f"⚠️ KVCacheManager store_cache失败，使用原有实现: {e}")

        # 原有的实现作为fallback
        # shape: (s, b * num_attention_heads, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()
        
        # 在store_cache中添加这个检查
        if torch.isnan(k_new.data).any() or torch.isnan(v_new.data).any():
            print(f"Replacing NaN values in cache at position {i}")
            k_new.data = torch.nan_to_num(k_new.data, nan=0.0, posinf=1.0, neginf=-1.0)
            v_new.data = torch.nan_to_num(v_new.data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        k_has_nan = torch.isnan(k_new.data).any()
        k_has_inf = torch.isinf(k_new.data).any()
        v_has_nan = torch.isnan(v_new.data).any()
        v_has_inf = torch.isinf(v_new.data).any()
        
        print(f"store_cache[{i}]: k_new NaN={k_has_nan}, Inf={k_has_inf}, v_new NaN={v_has_nan}, Inf={v_has_inf}")
        
        if k_has_nan or k_has_inf or v_has_nan or v_has_inf:
            print(f"❌ WARNING: Storing invalid data at position {i}!")
            print(f"k_new stats: mean={k_new.data.mean()}, std={k_new.data.std()}, min={k_new.data.min()}, max={k_new.data.max()}")
            print(f"v_new stats: mean={v_new.data.mean()}, std={v_new.data.std()}, min={v_new.data.min()}, max={v_new.data.max()}")
            
            # 可选：用零替换NaN/Inf
            # k_new.data = torch.where(torch.isnan(k_new.data) | torch.isinf(k_new.data), 
            #                         torch.zeros_like(k_new.data), k_new.data)
            # v_new.data = torch.where(torch.isnan(v_new.data) | torch.isinf(v_new.data), 
            #                         torch.zeros_like(v_new.data), v_new.data)
        
        print(f"store_cache, i: {i}, pos: {i}, k_new.shape[0]: {k_new.shape[0]}, k_new.shape[1]: {k_new.shape[1]}")
        # if i == self.task.gen_len - 1:  # last token, no need to store cache
        #     return
        print(f"store_cache, k_new: {k_new}, v_new: {v_new}")
        # if i == 0:  # prefill
        #     indices = (slice(0, k_new.shape[0]),
        #                slice(0, k_new.shape[1]))
        # else:  # decoding
        #     pos = i
        #     indices = (slice(pos, pos + k_new.shape[0]),
        #                slice(0, k_new.shape[1]))
        indices = (
            slice(i, i + k_new.shape[0]),    # start=i, end=i+new_len
            slice(0, k_new.shape[1])
        )

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)
        torch.cuda.synchronize()
        
        stored_k = k_home.data[indices[0], indices[1]]
        stored_v = v_home.data[indices[0], indices[1]]
        k_stored_nan = torch.isnan(stored_k).any()
        v_stored_nan = torch.isnan(stored_v).any()
        print(f"After store[{i}]: k_stored NaN={k_stored_nan}, v_stored NaN={v_stored_nan}")

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(
        self,
        hidden,
        cache_read_buf,
        weight_read_buf,
        attention_mask,
        cache_write_buf,
        i,
        k
    ):
        
        # num_attention_heads = self.config.num_attention_heads
        num_attention_heads = self.config.num_attention_heads
        
        # 🔧 添加forward开始调试信息
        offload_logger.info(f"FLEX_LlamaAttention.forward开始:")
        offload_logger.info(f"   - layer_id: {self.layer_id}")
        offload_logger.info(f"   - position: {i}")
        offload_logger.info(f"   - batch: {k}")
        offload_logger.info(f"   - cache_manager可用: {self.cache_manager is not None}")
        offload_logger.info(f"   - 当前设备: {hidden.val.device if hasattr(hidden, 'val') else 'Unknown'}")

        donate = [False] * 16
        h, donate[0] = hidden.val, True

        # k is batch index
        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (w_k, donate[4]), (w_v, donate[6]), (w_out, donate[8]), (input_layernorm, donate[10]), (rotary_emb_inv_freq, donate[12])) \
                = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _),  (w_v, _), (w_out, _), (input_layernorm, _), (rotary_emb_inv_freq, _)) = weight_read_buf.val

        print(f"attention forward, i: {i}, self.task.prompt_len: {self.task.prompt_len}, hiden states: {h}")
        
        if i == 0:
            # prefill
            # import pdb;pdb.set_trace()---------------------
            # see_memory_usage("-----------------------------------------before mha_llama ")
            print(f"attention forward, attention_mask: {attention_mask.val}")
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            print(f"attention forward, mask: {mask.data}")
            
            # 🔧 添加prefill调试信息
            offload_logger.info(f" Prefill阶段:")
            offload_logger.info(f"   - 使用mha_llama")
            offload_logger.info(f"   - mask设备: {mask.device}")
            offload_logger.info(f"   - compute设备: {self.compute}")
            
            h, new_k_cache, new_v_cache = self.compute.mha_llama(h, mask, w_q, w_k, w_v, w_out,
                                       num_attention_heads, donate, self.policy.compress_cache, self.policy.comp_cache_config, input_layernorm, rotary_emb_inv_freq)
            cache_write_buf.store((new_k_cache, new_v_cache))
            
            # 🔧 添加prefill结果调试信息
            offload_logger.info(f"Prefill完成:")
            offload_logger.info(f"   - new_k_cache形状: {new_k_cache.shape if hasattr(new_k_cache, 'shape') else 'Unknown'}")
            offload_logger.info(f"   - new_v_cache形状: {new_v_cache.shape if hasattr(new_v_cache, 'shape') else 'Unknown'}")
            offload_logger.info(f"   - new_k_cache设备: {new_k_cache.device if hasattr(new_k_cache, 'device') else 'Unknown'}")
            offload_logger.info(f"   - new_v_cache设备: {new_v_cache.device if hasattr(new_v_cache, 'device') else 'Unknown'}")
            
            # see_memory_usage("-----------------------------------------after mha_llama ")
        else:
            # decoding
            # see_memory_usage("-----------------------------------------before mha_gen_llama ")
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            print(f"attention forward, mask: {mask.data}, ")
            k_cache, v_cache = cache_read_buf.pop()
            # Optional memory logging
            # see_memory_usage("attention forward (decoding) before mha_gen_llama")

            if self.attention_compute == self.env.gpu:
                print(f"attention_compute == gpu")
            elif self.attention_compute == self.env.cpu:
                print(f"attention_compute == cpu")
            else:
                print(f"attention_compute == {self.attention_compute}")
            # k_cache = TorchTensor.create_from_torch(k_tensor, self.attention_compute)
            # v_cache = TorchTensor.create_from_torch(v_tensor, self.attention_compute)
            print(f"k_cache: {k_cache.shape}, self.policy.compress_cache: {self.policy.compress_cache}")
            
            # 🔧 添加decoding调试信息
            offload_logger.info(f" Decoding阶段:")
            offload_logger.info(f"   - 使用mha_gen_llama")
            offload_logger.info(f"   - k_cache形状: {k_cache.shape if hasattr(k_cache, 'shape') else 'Unknown'}")
            offload_logger.info(f"   - v_cache形状: {v_cache.shape if hasattr(v_cache, 'shape') else 'Unknown'}")
            offload_logger.info(f"   - k_cache设备: {k_cache.device if hasattr(k_cache, 'device') else 'Unknown'}")
            offload_logger.info(f"   - v_cache设备: {v_cache.device if hasattr(v_cache, 'device') else 'Unknown'}")
            offload_logger.info(f"   - attention_compute: {self.attention_compute}")
            offload_logger.info(f"   - compress_cache: {self.policy.compress_cache}")
            
            h, new_k_cache, new_v_cache = self.compute.mha_gen_llama(
                h, mask, w_q,
                w_k, w_v, w_out, num_attention_heads,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,
                input_layernorm,
                rotary_emb_inv_freq)
            print(f"attention forward, hiden state: {h}, new_k_cache: {new_k_cache}, new_v_cache: {new_v_cache}")
            cache_write_buf.store((new_k_cache, new_v_cache))
            
            # 🔧 添加decoding结果调试信息
            offload_logger.info(f" Decoding完成:")
            offload_logger.info(f"   - new_k_cache形状: {new_k_cache.shape if hasattr(new_k_cache, 'shape') else 'Unknown'}")
            offload_logger.info(f"   - new_v_cache形状: {new_v_cache.shape if hasattr(new_v_cache, 'shape') else 'Unknown'}")
            offload_logger.info(f"   - new_k_cache设备: {new_k_cache.device if hasattr(new_k_cache, 'device') else 'Unknown'}")
            offload_logger.info(f"   - new_v_cache设备: {new_v_cache.device if hasattr(new_v_cache, 'device') else 'Unknown'}")
            
            # see_memory_usage("-----------------------------------------after mha_gen_llama ")
        hidden.val = h
        self.temp_hidden_states.val=h
        
        # 🔧 添加forward完成调试信息
        offload_logger.info(f" FLEX_LlamaAttention.forward完成:")
        offload_logger.info(f"   - 输出hidden_states形状: {h.shape if hasattr(h, 'shape') else 'Unknown'}")
        offload_logger.info(f"   - 输出hidden_states设备: {h.device if hasattr(h, 'device') else 'Unknown'}")
        
        return h

class FLEX_LlamaMLP(LlamaMLP):
    def __init__(
        self,
        config: LlamaConfig,
        env: ExecutionEnv,
        policy: Policy,
        layer_id: int,
    ):
        super().__init__(config)
        self.config = config
        self.llama_config = get_llama_config('huggyllama/llama-7b')
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
                                else self.compute)

        self.task = None
        self.temp_hidden_states = ValueHolder()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        intermediate_size, h, dtype = (self.config.intermediate_size, self.config.hidden_size, np.float16)
        print('intermediate_size, h, dtype ', intermediate_size, h, dtype)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # 4 weight files
            # gate_proj
            ((intermediate_size, h), dtype, path + "mlp.gate_proj.weight"),
            # down_proj
            ((h, intermediate_size), dtype, path + "mlp.down_proj.weight"),
            # up_proj
            ((intermediate_size, h), dtype, path + "mlp.up_proj.weight"),
            # post attention layer norm
            ((h, ), dtype, path + "post_attention_layernorm.weight"),
        ]
        # see_memory_usage("-----------------------------------------before init weights of LLamaMLP ")
        weights = init_weight_list(weight_specs, self.policy, self.env)
        # see_memory_usage("-----------------------------------------after init weights of LLamaMLP ")
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        gate, down, up, post_attention_layernorm = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                    gate.smart_copy(dst1),
                    down.smart_copy(dst1),
                    up.smart_copy(dst1),
                    post_attention_layernorm.smart_copy(dst2)
            ))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, 
        hidden_states,
        cache_read_buf,
        weight_read_buf,
        attention_mask,
        cache_write_buf,
        position_ids,
        k: int = 0,
        generated_tokens_num: int = 0,
        ):
        donate = [False] * 9
        h, donate[0] = hidden_states.val, True
        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((gate, donate[1]), (down, donate[3]),
             (up, donate[5]), (post_attention_layernorm, donate[7])) = weight_read_buf.pop()
        else:
            ((gate, _), (down, _),
             (up, _), (post_attention_layernorm, _)) = weight_read_buf.val

        h = self.compute.mlp_llama(h, gate, down, up, donate, self.config, post_attention_layernorm)
        hidden_states.val = h
        self.temp_hidden_states.val=h
        
        return h



# class OutputEmbed:
#     def __init__(self, config, env, policy):
#         self.config = config
#         self.env = env
#         self.policy = policy
#         self.compute = self.env.gpu
#         self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
#             else self.compute)

#         self.task = None

#     def set_task(self, task):
#         self.task = task

#     def init_weight(self, weight_home, path):
#         v, h, dtype = (self.config.vocab_size, self.config.hidden_size,
#             self.config.dtype)
#         path = os.path.join(path, "")
#         weight_specs = [
#             # w_ln
#             ((h,), dtype, path + "norm.weight"),
#             # lm_head.weight
#             ((v, h), dtype, path + "lm_head.weight"),
#             # ((v, h), dtype, path + "embed_tokens.weight"),
#         ]
#         weights = init_weight_list(weight_specs, self.policy, self.env)

#         weight_home.store(weights)

#     def load_weight(self, weight_home, weight_read_buf, k):
#         w_ln, lm_head = weight_home.val
#         if k == 0:
#             dst1 = self.weight_load_dst
#             dst2 = self.compute
#             weight_read_buf.store((w_ln.smart_copy(dst2),
#                 lm_head.smart_copy(dst1)))

#     def init_cache_one_gpu_batch(self, cache_home):
#         pass  # do nothing

#     def load_cache(self, cache_home, cache_read_buf, i):
#         pass  # do nothing

#     def store_cache(self, cache_home, cache_write_buf, i):
#         pass  # do nothing

#     def input_act_shape_and_dtype(self, batch_size, seq_len):
#         return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

#     def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
#                 cache_write_buf, i, k):
#         donate = [False] * 4
#         h, donate[0] = hidden.val, True

#         if k == self.policy.num_gpu_batches - 1:
#             # Clear the weight_read_buf if it is the last gpu batch
#             (w_ln, donate[1]), (lm_head, donate[3]) = weight_read_buf.pop()
#         else:
#             (w_ln, _), (lm_head, _) = weight_read_buf.val

#         h = self.compute.llama_output_embed(h, w_ln, donate,
#             self.task.do_sample, self.task.temperature, lm_head, self.task.top_p)
#         hidden.val = h

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        
        self.self_attn = FLEX_LlamaAttention(config=config, env=env, policy=policy, layer_id=layer_id)
        self.mlp = FLEX_LlamaMLP(
            layer_id=layer_id,
            env=env,
            policy=policy,
            config=config
        )
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.compute = self.self_attn.compute
        self.policy = policy

    def set_task(self, task):
        self.self_attn.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.self_attn.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.self_attn.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
           weight_read_buf.store((read_buf1, read_buf2))

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            offload_logger.info(f" Prefill阶段，跳过load_cache - 位置:{i}, 层:{j}, 批次:{k}")
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        offload_logger.info(f"LlamaDecoderLayer.load_cache - 位置:{i}, 层:{j}, 批次:{k}")
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            offload_logger.info(f"最后一个token，跳过store_cache - 位置:{i}, 层:{j}, 批次:{k}")
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        offload_logger.info(f" LlamaDecoderLayer.store_cache - 位置:{i}, 层:{j}, 批次:{k}")
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos - 1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j - 1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos + 1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos + 1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        

        if i > 0:  # 只在decoding阶段调用load_cache
            offload_logger.info(f" 调用load_cache - 位置:{i}, 层:{j}, 批次:{k}")
            self.load_cache(i, j, k, overlap=False)
        
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                               self.weight_read_buf[j], self.attention_mask[k],
                               self.cache_write_buf[j][k], i, k)
        

        offload_logger.info(f" 调用store_cache - 位置:{i}, 层:{j}, 批次:{k}")
        self.store_cache(i, j, k, overlap=False)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(
        self,
        inputs,
        max_new_tokens: int=32,
        do_sample: bool=True,
        temperature: float=0.6,
        stop: Optional[int] = None,
        debug_mode: Optional[str] = None,
        cut_gen_len: Optional[int] = None,
        top_p: float = 0.9,
        verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            top_p=top_p
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename

def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = [
        # "Simply put, the theory of relativity states that ",

        "I believe the meaning of life is",

        # """Translate English to French:
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
    ]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts

def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = '[PAD]'
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
    # Task and policy
    # warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    ## weight and cache compression
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"
    llama_config = get_llama_config(args.model)
    cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = LlamaLM(llama_config, env, args.path, policy)
    try:
        # print("warmup - generate")
        # output_ids = model.generate(
        #     warmup_inputs,
        #     max_new_tokens=1,
        #     debug_mode=args.debug_mode,
        #     verbose=args.verbose)
        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs,
            max_new_tokens=gen_len,
            debug_mode=args.debug_mode,
            cut_gen_len=cut_gen_len, 
            verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        llama_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="huggyllama/llama-7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="/tmp/data/llama_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="/tmp/data/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[50, 50, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
