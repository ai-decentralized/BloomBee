"""
Tools for converting transformer blocks, applying quantization and/or tensor parallelism
"""
import re
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import tensor_parallel as tp
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from tensor_parallel.slicing_configs import get_bloom_config
from transformers import PretrainedConfig
from pynvml import *
from bloombee.utils.debug import dprint
from bloombee.utils.memory_usage import see_memory_usage, log_mem
from bloombee.server.tensor_parallel import LlamaTensorParallelAdapter

logger = get_logger(__name__)


def _get_choice(cur_percent, percents, choices):
    """Return which device a parameter belongs to based on its cumulative position.

    Mirrors LLaMA's get_choice in flex_llama.py / from_pretrained.py.

    Args:
        cur_percent: Midpoint percentage (0-100) of this parameter in the full model.
        percents:    Allocation percentages [disk%, cpu%, gpu%] that must sum to 100.
        choices:     Corresponding device choices.
    """
    cum = np.cumsum(percents)
    for i, boundary in enumerate(cum):
        if cur_percent < boundary:
            return choices[i]
    return choices[-1]


def _assign_param_devices(module, policy, gpu_device):
    """Assign each named parameter to CPU or GPU using the same cumulative-midpoint
    logic that LLaMA's FlexGen system uses (init_weight_list in flex_llama.py).

    With policy.w_gpu_percent=50 / w_cpu_percent=50 the first ~50% of parameters
    (by element count) are placed on CPU, the remaining ~50% on GPU.

    Disk offload is not supported for standard HF modules; w_disk_percent is merged
    into the CPU allocation instead.

    Returns:
        dict mapping parameter name → torch.device
    """
    cpu_device = torch.device('cpu')

    param_list = list(module.named_parameters())
    if not param_list:
        return {}

    sizes = np.array([p.numel() for _, p in param_list], dtype=np.float64)
    sizes_cumsum = np.cumsum(sizes)
    total = sizes_cumsum[-1]

    # Merge disk% into CPU% (disk offload not implemented for HF blocks)
    effective_cpu = getattr(policy, 'w_cpu_percent', 0) + getattr(policy, 'w_disk_percent', 0)
    effective_gpu = getattr(policy, 'w_gpu_percent', 100)
    # percents must sum to 100; the first bucket (0%) is a placeholder so that the
    # cumulative sum aligns with [effective_cpu, 100] boundaries.
    dev_percents = [0.0, float(effective_cpu), float(effective_gpu)]
    dev_choices  = [cpu_device, cpu_device, gpu_device]

    param_devices = {}
    for i, (name, _) in enumerate(param_list):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / total * 100
        param_devices[name] = _get_choice(mid_percent, dev_percents, dev_choices)
    return param_devices


class QuantType(Enum):
    """
    Quantization type enum for FlexGen compression.
    Note: bitsandbytes quantization is not used. This enum only controls FlexGen's group-wise quantization.
    """
    NONE = 0
    INT8 = 1  # 8-bit group-wise quantization for FlexGen
    NF4 = 2  # 4-bit group-wise quantization for FlexGen


def convert_block(
    block: nn.Module,
    block_index: int,
    config: PretrainedConfig,
    tensor_parallel_devices: Sequence[torch.device],
    output_device: torch.device,
    quant_type: QuantType,
    freeze: bool = True,
    adapters: Optional[Sequence[str]] = None,
    policy=None,
    **kwargs,
) -> tp.TensorParallel:
    """
    Optimize a transformer block for use in a Petals server with FlexGen.
    
    Note: Quantization is handled by FlexGen's weight loading system, not here.
    The quant_type parameter is passed through but not used in this function.

    :note: some optimizations will modify the input block in-place!
    :param block: a single transformer block, either pre-trained or newly initialized
    :param config: HF transformers config for the full model
    :param tensor_parallel_devices: if specified, use tensor parallelism to split the model between these devices
    :note: if there is only a single device, model wil still be wrapped with TensorParallel (for uniformity)
    :param output_device: if tensor_parallel_devices is True, output
    :param quant_type: quantization type (used by FlexGen compression, not applied here)
    :param freeze: if True (default), make all module parameters non-trainable
    :return: a module that acts like the original block, but runs with all specified optimizations

    """
    if freeze:
        block.requires_grad_(False)
    if len(tensor_parallel_devices) > 1 and config.model_type == "llama":
        return make_tensor_parallel(block, config, tensor_parallel_devices, output_device, policy=policy)

    # Skip tensor parallelism for FlexGen blocks - they manage their own weights and devices
    log_prefix = f"[convert_block:{block_index}]"
    # log_mem(f"{log_prefix} skipping tensor parallelism - FlexGen manages weights directly")
    
    # Quantization is handled by FlexGen's compression system during weight loading
    # No bitsandbytes quantization is applied here
    # log_mem(f"{log_prefix} quantization handled by FlexGen compression system")
    
    # Create a simple wrapper that provides TensorParallel interface for pipeline parallelism
    # but uses FlexGen's forward method directly
    class PipelineParallelWrapper:
        def __init__(self, module, devices, output_device, block_index=0, policy=None):
            self._module = module
            self.devices = devices
            self.output_device = output_device
            self.output_device_index = 0  # Single device in pipeline parallelism
            self.module_shards = [module]  # Single shard per pipeline stage

            # Fine-grained per-parameter CPU/GPU split, mirroring LLaMA's FlexGen approach.
            # FlexGen blocks use meta-device initially — skip them.
            first_param = next(iter(module.parameters()), None)
            is_hf_block = (
                first_param is not None
                and first_param.device.type != 'meta'
                and output_device is not None
            )

            self._param_devices = {}   # name → torch.device
            self._cpu_offload = False

            if is_hf_block and policy is not None:
                effective_cpu = (
                    getattr(policy, 'w_cpu_percent', 0)
                    + getattr(policy, 'w_disk_percent', 0)
                )
                if effective_cpu > 0:
                    # Assign each parameter individually using cumulative-midpoint logic.
                    # Buffers (e.g. rotary embedding inv_freq, lazy cos_cached/sin_cached)
                    # are kept on GPU at all times to avoid device-mismatch issues with
                    # lazily-registered buffers (registered during the first forward call).
                    self._param_devices = _assign_param_devices(module, policy, output_device)
                    pin = getattr(policy, 'pin_weight', False) and output_device.type == 'cuda'

                    # Move parameters to their assigned device
                    for name, param in module.named_parameters():
                        target = self._param_devices[name]
                        if target.type == 'cpu':
                            if pin:
                                param.data = param.data.cpu().pin_memory()
                            else:
                                param.data = param.data.cpu()
                        else:
                            param.data = param.data.to(output_device)

                    # Always keep buffers on GPU so that lazily-registered buffers
                    # (like Falcon's cos_cached / sin_cached) are created on GPU too.
                    for buf_name, buf in list(module.named_buffers()):
                        if buf is not None and buf.device.type != output_device.type:
                            # Navigate to the submodule that owns this buffer and re-register
                            parts = buf_name.split('.')
                            submod = module
                            for part in parts[:-1]:
                                submod = getattr(submod, part)
                            submod.register_buffer(parts[-1], buf.to(output_device), persistent=False)

                    self._cpu_offload = any(
                        d.type == 'cpu' for d in self._param_devices.values()
                    )
                    n_cpu = sum(1 for d in self._param_devices.values() if d.type == 'cpu')
                    n_gpu = len(self._param_devices) - n_cpu
                    logger.info(
                        f"[block {block_index}] Per-parameter CPU offload: "
                        f"{n_cpu}/{len(self._param_devices)} params on CPU, "
                        f"{n_gpu}/{len(self._param_devices)} params on GPU "
                        f"(w_gpu={getattr(policy,'w_gpu_percent',100)}%, "
                        f"w_cpu={getattr(policy,'w_cpu_percent',0)}%)"
                    )
                else:
                    # All weights on GPU
                    module.to(output_device)
            elif is_hf_block:
                module.to(output_device)

        def forward(self, *args, **kwargs):
            if self._cpu_offload:
                # Move CPU-resident parameters to GPU before forward.
                # Buffers stay on GPU permanently (see __init__).
                for name, param in self._module.named_parameters():
                    if self._param_devices.get(name, self.output_device).type == 'cpu':
                        param.data = param.data.to(self.output_device, non_blocking=True)
                if self.output_device.type == 'cuda':
                    torch.cuda.synchronize(self.output_device)

                result = self._module.forward(*args, **kwargs)

                # Restore CPU params asynchronously after forward
                for name, param in self._module.named_parameters():
                    if self._param_devices.get(name, self.output_device).type == 'cpu':
                        param.data = param.data.to('cpu', non_blocking=True)
                return result
            return self._module.forward(*args, **kwargs)
            
        def parameters(self, *args, **kwargs):
            return self._module.parameters(*args, **kwargs)

        def named_parameters(self, *args, **kwargs):
            return self._module.named_parameters(*args, **kwargs)

        def parameters(self, *args, **kwargs):
            return self._module.parameters(*args, **kwargs)
            
        def named_buffers(self, *args, **kwargs):
            return self._module.named_buffers(*args, **kwargs)

        def buffers(self, *args, **kwargs):
            return self._module.buffers(*args, **kwargs)
        
        def rms_norm(self, *args, **kwargs):
            if hasattr(self._module, 'rms_norm'):
                return self._module.rms_norm(*args, **kwargs)
            return None

        def load_lm_head(self, *args, **kwargs):
            if hasattr(self._module, 'load_lm_head'):
                return self._module.load_lm_head(*args, **kwargs)
            # No-op for non-FlexGen models (Falcon, Mixtral)

        def lm_head_forward(self, *args, **kwargs):
            if hasattr(self._module, 'lm_head_forward'):
                return self._module.lm_head_forward(*args, **kwargs)
            return None
    
    tp_block = PipelineParallelWrapper(block, tensor_parallel_devices, output_device, block_index=block_index, policy=policy)
    # log_mem(f"{log_prefix} created PipelineParallel wrapper")
    
    dprint('quant_type ', quant_type)
    dprint('adapters ', adapters )
    if adapters:
        
        from bloombee.utils.peft import add_adapter_to_block, create_lora_adapter, load_peft

        create_lora_adapter(tp_block)
        for adapter_name in adapters:
            adapter_config, adapter_state_dict = load_peft(
                adapter_name,
                block_idx=block_index,
                **kwargs,
            )
            add_adapter_to_block(tp_block, block_index, adapter_name, adapter_config, adapter_state_dict)

    return tp_block


# NOTE: bitsandbytes quantization has been removed.
# Quantization is now handled entirely by FlexGen's compression system during weight loading.
# This function is kept for backward compatibility but does nothing.
def quantize_module(model: nn.Module, *, quant_type: QuantType) -> nn.Module:
    """
    Deprecated: Quantization is now handled by FlexGen's compression system.
    This function is a no-op and kept for backward compatibility.
    """
    if quant_type != QuantType.NONE:
        logger.debug(f"Quantization type {quant_type} specified, but quantization is handled by FlexGen compression system")
    return model


def make_tensor_parallel(
    block: nn.Module,
    model_config: PretrainedConfig,
    devices: Sequence[torch.device],
    output_device: torch.device,
    policy=None,
) -> nn.Module:
    if model_config.model_type == "llama":
        tp_block = tp.TensorParallel(block, devices, config=None, output_device=output_device, delay_init=True)
        return LlamaTensorParallelAdapter(
            tp_block,
            model_config,
            layer_idx=getattr(block, "layer_idx", 0),
            policy=policy,
        )

    if model_config.model_type == "bloom":
        tp_config = get_bloom_config(model_config, devices)
        del tp_config.state_rules[re.compile(".*word_embeddings.weight$")]
    else:
        if len(devices) > 1:
            logger.warning("Tensor parallelism is not tested for models other than BLOOM yet, proceed with caution")
        tp_config = None
    tp_block = tp.TensorParallel(block, devices, config=tp_config, output_device=output_device, delay_init=True)
    # print('make_tensor_parallel: tp_block ', tp_block)
    # import pdb; pdb.set_trace()
    total_heads = 0
    for tp_shard in tp_block.module_shards:
        for submodule in tp_shard.modules():
            # print("flex_llama.LlamaAttention ", flex_llama.LlamaAttention)
            # print("submodule ", submodule)
            if isinstance(submodule, model_config.attn_class):
                total_heads += submodule.num_heads
    if model_config.model_type == "bloom":
        assert total_heads == model_config.num_attention_heads
    return tp_block


def check_device_balance(devices: Sequence[torch.device]):
    if not all(device.type == "cuda" for device in devices):
        logger.warning("Running tensor parallelism on non-GPU devices; proceed at your own risk")
        return
    unique_device_capabilities = set(map(torch.cuda.get_device_capability, devices))
    if len(unique_device_capabilities) > 1:
        logger.warning(
            f"Found GPUs with uneven capabilities: {unique_device_capabilities}. "
            f"Using GPUs with different performance will cause the server to wait for the slowest GPU."
        )

    memory_per_device = tuple(torch.cuda.get_device_properties(device).total_memory for device in devices)
    used_memory = min(memory_per_device) * len(memory_per_device)
    wasted_memory_rate = (sum(memory_per_device) - used_memory) / sum(memory_per_device)
    if wasted_memory_rate > 0.05:
        logger.warning(
            f"GPU devices have highly uneven memory, {wasted_memory_rate * 100:.2f}% memory is wasted. "
            f"Consider running high-memory GPUs in a separate server."
        )
