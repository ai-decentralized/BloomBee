from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel

from bloombee.models.mixtral.block import WrappedMixtralBlock
from bloombee.utils.convert_block import QuantType
from bloombee.utils.misc import get_size_in_bytes
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import fix_recursive_import
from bloombee.flexgen_utils.utils import ValueHolder, array_1d


def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it using BloomConfig. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    if config.torch_dtype not in ("auto", None, torch.float32):
        # If config specifies float32, we override it to the default dtype below
        return config.torch_dtype
    return torch.bfloat16


def get_block_size(
    config: PretrainedConfig,
    location: str,
    env: ExecutionEnv,
    policy: Policy,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    quant_type: QuantType = QuantType.NONE,
    eps: float = 0.01,  # eps accounts for ~1% of metainfo for tensor descriptions, quantization tables, etc.
) -> int:
    if location == "memory":
        assert (
            dtype is not None and quant_type is not None
        ), 'get_block_size(..., location="memory") requires to specify dtype and quant_type for calculations'

    with init_empty_weights(include_buffers=False):
        block = get_model_block(config, env, policy)
        n_params = sum(param.numel() for param in block.parameters())

    if location == "memory":
        if quant_type == QuantType.NONE:
            dtype = resolve_block_dtype(config, dtype)
            bytes_per_value = get_size_in_bytes(dtype)
        elif quant_type == QuantType.INT8:
            bytes_per_value = 1
        elif quant_type == QuantType.NF4:
            bytes_per_value = 4.25 / 8  # Bitness of NF4 with this config (measured empirically)
        else:
            raise ValueError(f"Unsupported quant_type={quant_type}")
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)

    return round(n_params * bytes_per_value * (1 + eps))


def get_model_block(config, env, policy, weight_home, path, layer_idx: int = 0):
    """
    The function to create a model block based on the block class
    kwargs argument **only** is necessary for specific classes, like Mixtral.
    They will not be passed to other block constructors.
    """
    if config.block_class == WrappedMixtralBlock:
        print('server/block_utils.py config.block_class == WrappedMixtralBlock ')
        config = PreTrainedModel._autoset_attn_implementation(config)
        return config.block_class(config, layer_idx)
    # config.block_class == WrappedLlamaBlock in distributedllamaconfig in config.py
    # print('server/block_utils.py get_model_block() : config', config)
    res = config.block_class(config, layer_idx, env, policy, weight_home, path)  # go to block.py class OptimizedLlamaDecoderLayer
    # print(' get_model_block res  ', res)
    return res  # res is only nn.module without weights
