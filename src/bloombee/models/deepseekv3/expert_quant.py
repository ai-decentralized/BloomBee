"""Symmetric group-wise int8 quantization for DeepSeek-V3's batched expert weights.

DeepseekV3Experts (transformers.models.deepseek_v3.modeling_deepseek_v3) stores every
routed expert's weights as two 3D nn.Parameter tensors -- gate_up_proj and down_proj,
shaped [num_experts, out_features, in_features] -- rather than per-expert nn.Linear
submodules. That rules out bitsandbytes' Linear8bitLt/Linear4bit, which replace
nn.Linear instances; there's nothing of that shape to attach to here. These functions
quantize/dequantize the raw tensors directly instead.

Quantization is per (expert, output-row, group-of-`group_size`-input-channels),
symmetric int8 (range [-127, 127]), matching the row/group-wise scheme most GPTQ/AWQ
style quantizers use for the input (contraction) dimension of a matmul.
"""
from typing import Tuple

import torch


def _effective_group_size(in_features: int, group_size: int) -> int:
    """Fall back to whole-row quantization if in_features doesn't divide evenly.

    Real DeepSeek-V3 (hidden_size=7168, moe_intermediate_size=2048) divides evenly by
    the default group_size=128, but tiny test configs used in unit tests won't.
    """
    if group_size <= 0 or in_features % group_size != 0:
        return in_features
    return group_size


def quantize_expert_weight(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize a [num_experts, out_features, in_features] weight tensor to int8.

    :param weight: full-precision weight tensor (e.g. gate_up_proj or down_proj)
    :param group_size: number of input channels sharing one scale factor
    :return: (data, scale, effective_group_size)
        data:  int8 tensor, same shape as `weight`
        scale: float32 tensor [num_experts, out_features, num_groups], on the weight device
    """
    num_experts, out_features, in_features = weight.shape
    eff_group_size = _effective_group_size(in_features, group_size)
    num_groups = in_features // eff_group_size

    grouped = weight.detach().reshape(num_experts, out_features, num_groups, eff_group_size)
    data = torch.empty_like(grouped, dtype=torch.int8)
    scale = torch.empty((num_experts, out_features, num_groups), device=weight.device, dtype=torch.float32)
    # Work one expert at a time to avoid a full-model-sized float32 temporary.
    for index in range(num_experts):
        values = grouped[index].float()
        expert_scale = values.abs().amax(dim=-1).clamp_min(1e-8) / 127.0
        scale[index].copy_(expert_scale)
        data[index].copy_((values / expert_scale.unsqueeze(-1)).round().clamp_(-127, 127))

    return data.reshape(num_experts, out_features, in_features).contiguous(), scale, eff_group_size


def dequantize_expert_weight(
    data: torch.Tensor,
    scale: torch.Tensor,
    expert_idx,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a single expert's weight matrix on demand.

    :param data: int8 tensor [num_experts, out_features, in_features]
    :param scale: tensor [num_experts, out_features, num_groups]
    :param expert_idx: index (int or 0-dim tensor) of the expert to dequantize
    :param group_size: the *effective* group size used at quantization time
    :param dtype: output dtype (should match the activation dtype for the matmul)
    :return: dequantized weight matrix [out_features, in_features]
    """
    d = data[expert_idx]  # [out_features, in_features] int8
    s = scale[expert_idx]  # [out_features, num_groups]
    out_features, in_features = d.shape
    num_groups = s.shape[-1]

    d = d.reshape(out_features, num_groups, group_size).float()
    d = d * s.unsqueeze(-1).float()
    return d.reshape(out_features, in_features).to(dtype)
