"""
Tools for converting transformer blocks, applying quantization and/or tensor parallelism
"""
import re
from enum import Enum
from typing import Optional, Sequence

import tensor_parallel as tp
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from tensor_parallel.slicing_configs import get_bloom_config
from transformers import PretrainedConfig
from pynvml import *
from bloombee.utils.memory_usage import see_memory_usage, log_mem

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

logger = get_logger(__name__)
use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


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
    # Skip tensor parallelism for FlexGen blocks - they manage their own weights and devices
    log_prefix = f"[convert_block:{block_index}]"
    # log_mem(f"{log_prefix} skipping tensor parallelism - FlexGen manages weights directly")
    
    # Quantization is handled by FlexGen's compression system during weight loading
    # No bitsandbytes quantization is applied here
    # log_mem(f"{log_prefix} quantization handled by FlexGen compression system")
    
    # Create a simple wrapper that provides TensorParallel interface for pipeline parallelism
    # but uses FlexGen's forward method directly
    class PipelineParallelWrapper:
        def __init__(self, module, devices, output_device):
            self._module = module
            self.devices = devices
            self.output_device_index = 0  # Single device in pipeline parallelism
            self.module_shards = [module]  # Single shard per pipeline stage
            
        def forward(self, *args, **kwargs):
            # Use FlexGen's forward method directly
            return self._module.forward(*args, **kwargs)
            
        def named_parameters(self, *args, **kwargs):
            return self._module.named_parameters(*args, **kwargs)
            
        def named_buffers(self, *args, **kwargs):
            return self._module.named_buffers(*args, **kwargs)
        
        def rms_norm(self, *args, **kwargs):
            return self._module.rms_norm(*args, **kwargs)
        
        def load_lm_head(self, *args, **kwargs):
            return self._module.load_lm_head(*args, **kwargs)
        
        def lm_head_forward(self, *args, **kwargs):
            return self._module.lm_head_forward(*args, **kwargs)
    
    tp_block = PipelineParallelWrapper(block, tensor_parallel_devices, output_device)
    # log_mem(f"{log_prefix} created PipelineParallel wrapper")
    
    print('quant_type ', quant_type)
    print('adapters ', adapters )
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
    block: nn.Module, model_config: PretrainedConfig, devices: Sequence[torch.device], output_device: torch.device
) -> nn.Module:
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
