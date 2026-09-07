from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel

from bloombee.models.bloom.block import WrappedBloomBlock
from bloombee.models.deepseekv3.block import WrappedDeepseekV3Block
from bloombee.models.falcon.block import WrappedFalconBlock
from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.models.mixtral.block import WrappedMixtralBlock
from bloombee.models.qwen3.block import WrappedQwen3Block
from bloombee.utils.convert_block import QuantType
from bloombee.utils.misc import get_size_in_bytes
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import fix_recursive_import
from bloombee.flexgen_utils.utils import ValueHolder, array_1d
from bloombee.utils.debug import dprint


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
        dummy_weight_home = array_1d(2, ValueHolder)
        # skip_init_weights: this block exists only to count parameters. Without
        # it, the FlexGen llama path ran a full weight download + numpy
        # conversion (13 GB for llama-7b) into the dummy "/tmp" path just to
        # estimate block size.
        #
        # Block param count can depend on layer_idx: e.g. DeepSeek-V3's
        # first_k_dense_replace makes early layers a single dense MLP while
        # later layers hold all routed+shared experts (~20x the params). A
        # single-layer_idx sample (previously always layer_idx=0, i.e. always
        # dense) would badly undercount the memory Server._choose_num_blocks
        # reserves per block, risking OOM once a heavier block is assigned.
        # Sample every layer position and keep the largest so capacity
        # planning reflects the worst-case block this config can produce.
        #
        # Also track how many of those params live in DeepSeek-V3's batched
        # expert tensors (block.mlp.experts.{gate_up_proj,down_proj}), since
        # convert_block() quantizes only those to int8 -- everything else in
        # the block (attention, router, dense MLP layers) stays at `dtype`.
        n_params = 0
        n_expert_params = 0
        for layer_idx in range(getattr(config, "num_hidden_layers", 1)):
            block = get_model_block(
                config, env, policy, dummy_weight_home, "/tmp", layer_idx=layer_idx, skip_init_weights=True
            )
            total = sum(param.numel() for param in block.parameters())
            if total < n_params:
                continue
            experts = getattr(getattr(block, "mlp", None), "experts", None)
            expert_params = 0
            if experts is not None and hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
                expert_params = experts.gate_up_proj.numel() + experts.down_proj.numel()
            n_params, n_expert_params = total, expert_params

    if location == "memory":
        dtype = resolve_block_dtype(config, dtype)
        bytes_per_value = get_size_in_bytes(dtype)
        if quant_type == QuantType.NONE:
            return round(n_params * bytes_per_value * (1 + eps))
        if quant_type is not QuantType.INT8 or config.block_class is not WrappedDeepseekV3Block:
            raise ValueError(
                f"quant_type={quant_type} is not supported for block_class={config.block_class.__name__}; "
                "only QuantType.INT8 for DeepSeek-V3 is implemented (see convert_block())."
            )
        # Quantized experts: ~1 byte/param (int8) plus the group scale table, which
        # `eps` already accounts for at group_size=128 (~1/128 the tensor's element
        # count, at 2-4 bytes each -- well under the 1% eps budget). Everything else
        # in the block (attention, router, dense layers) stays at `bytes_per_value`.
        non_expert_params = n_params - n_expert_params
        quantized_bytes = n_expert_params * 1 + non_expert_params * bytes_per_value
        return round(quantized_bytes * (1 + eps))
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)
        return round(n_params * bytes_per_value * (1 + eps))


def _autoset_attn_impl(config):
    """Set ``config._attn_implementation`` in a way that works on TF 4.x and 5.x.

    TF 4.x exposed ``PreTrainedModel._autoset_attn_implementation(config)`` which
    picked sdpa/flash/eager based on availability and wrote it to the config.
    TF 5.x removed that classmethod; its replacement (``set_attn_implementation``)
    is an instance method on an already-built model, which is backwards for us —
    we need the decision *before* instantiation.

    For BloomBee's single-block use case, "eager" is always a safe choice:
    flash-attn isn't on our V100 stack, and sdpa requires a batched mask path
    we don't always provide. We only write eager if the caller hasn't pinned
    something else on the config already.
    """
    legacy = getattr(PreTrainedModel, "_autoset_attn_implementation", None)
    if legacy is not None:
        return legacy(config)
    if getattr(config, "_attn_implementation", None) in (None, "", "auto"):
        config._attn_implementation = "eager"
    return config


def get_model_block(config, env, policy, weight_home, path, layer_idx: int = 0, skip_init_weights: bool = False):
    """
    The function to create a model block based on the block class.
    - Bloom:   takes (config) only, no layer_idx, no FlexGen args
    - Mixtral: takes (config, layer_idx), no FlexGen args
    - Falcon:  takes (config) only, no layer_idx, no FlexGen args
    - Llama:   takes (config, layer_idx, env, policy, weight_home, path) — FlexGen-based
    """
    if config.block_class == WrappedBloomBlock:
        dprint('server/block_utils.py config.block_class == WrappedBloomBlock ')
        return config.block_class(config, layer_idx)
    if config.block_class == WrappedMixtralBlock:
        dprint('server/block_utils.py config.block_class == WrappedMixtralBlock ')
        config = _autoset_attn_impl(config)
        return config.block_class(config, layer_idx)
    elif config.block_class == WrappedFalconBlock:
        dprint('server/block_utils.py config.block_class == WrappedFalconBlock ')
        return config.block_class(config)
    elif config.block_class == WrappedQwen3Block:
        dprint('server/block_utils.py config.block_class == WrappedQwen3Block ')
        config = _autoset_attn_impl(config)
        return config.block_class(config, layer_idx)
    elif config.block_class == WrappedGemma4Block:
        dprint('server/block_utils.py config.block_class == WrappedGemma4Block ')
        config = _autoset_attn_impl(config)
        return config.block_class(config, layer_idx)
    elif config.block_class == WrappedDeepseekV3Block:
        dprint('server/block_utils.py config.block_class == WrappedDeepseekV3Block ')
        config = _autoset_attn_impl(config)
        return config.block_class(config, layer_idx)
    # config.block_class == WrappedLlamaBlock in distributedllamaconfig in config.py
    # print('server/block_utils.py get_model_block() : config', config)
    res = config.block_class(
        config, layer_idx, env, policy, weight_home, path, skip_init_weights=skip_init_weights
    )  # go to block.py class OptimizedLlamaDecoderLayer
    # print(' get_model_block res  ', res)
    return res  # res is only nn.module without weights
