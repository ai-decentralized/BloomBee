from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch


_QUANT_ENV = "BLOOMBEE_S2S_ACTIVATION_QUANT"
_SPEC_ONLY_ENV = "BLOOMBEE_S2S_ACTIVATION_QUANT_SPEC_ONLY"
_LOGS_ENV = "BLOOMBEE_S2S_ACTIVATION_QUANT_LOGS"


def s2s_activation_quant_enabled(*, is_spec_dec: bool) -> bool:
    mode = os.environ.get(_QUANT_ENV, "").strip().lower()
    if mode in ("", "0", "false", "off", "none"):
        return False
    if mode not in ("1", "true", "on", "int8", "int8_per_token"):
        return False
    spec_only = os.environ.get(_SPEC_ONLY_ENV, "1").strip().lower() not in ("0", "false", "off")
    return bool(is_spec_dec) or not spec_only


def quantize_s2s_hidden_for_transport(
    hidden_states: torch.Tensor,
    *,
    is_spec_dec: bool,
    logger: Optional[Any] = None,
    context: str = "",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    if not s2s_activation_quant_enabled(is_spec_dec=is_spec_dec):
        return hidden_states, None, None
    if not torch.is_floating_point(hidden_states) or hidden_states.ndim < 2:
        return hidden_states, None, None

    original_dtype = str(hidden_states.dtype).replace("torch.", "")
    original_shape = [int(dim) for dim in hidden_states.shape]
    source = hidden_states.detach()
    reduce_dim = source.ndim - 1
    max_abs = source.abs().float().amax(dim=reduce_dim, keepdim=True)
    scale = torch.clamp(max_abs / 127.0, min=1e-8)
    quantized = torch.round(source.float() / scale).clamp_(-127, 127).to(torch.int8).contiguous()

    scale_tensor = scale.squeeze(reduce_dim).float().contiguous()
    quant_meta: Dict[str, Any] = {
        "scheme": "int8_per_token",
        "orig_dtype": original_dtype,
        "orig_shape": original_shape,
        "scale_shape": [int(dim) for dim in scale_tensor.shape],
    }
    if logger is not None and os.environ.get(_LOGS_ENV, "0") == "1":
        raw_bytes = int(hidden_states.numel() * hidden_states.element_size())
        quant_bytes = int(quantized.numel() * quantized.element_size() + scale_tensor.numel() * scale_tensor.element_size())
        logger.info(
            "[S2S_ACTIVATION_QUANT] context=%s scheme=int8_per_token shape=%s raw_bytes=%s quant_bytes=%s ratio=%.4f",
            context,
            tuple(original_shape),
            raw_bytes,
            quant_bytes,
            quant_bytes / raw_bytes if raw_bytes > 0 else 1.0,
        )
    return quantized, scale_tensor, quant_meta


def dequantize_s2s_hidden_from_transport(
    hidden_states: torch.Tensor,
    metadata: Optional[Dict[str, Any]],
    scale_tensor: Optional[torch.Tensor] = None,
    *,
    logger: Optional[Any] = None,
    context: str = "",
) -> torch.Tensor:
    if not isinstance(metadata, dict):
        if hidden_states.dtype == torch.int8:
            raise ValueError("Received int8 S2S hidden states without quantization metadata")
        return hidden_states
    quant_meta = metadata.get("s2s_hidden_quant")
    if not isinstance(quant_meta, dict):
        if hidden_states.dtype == torch.int8:
            raise ValueError("Received int8 S2S hidden states without quantization metadata")
        return hidden_states
    if quant_meta.get("scheme") != "int8_per_token":
        if hidden_states.dtype == torch.int8:
            raise ValueError(f"Unsupported S2S activation quantization scheme: {quant_meta.get('scheme')!r}")
        return hidden_states
    if hidden_states.dtype != torch.int8:
        return hidden_states

    scale_shape = tuple(int(dim) for dim in quant_meta.get("scale_shape", ()))
    scale_values = quant_meta.get("scale")
    if not scale_shape:
        raise ValueError("Missing S2S activation quantization scale metadata")

    if scale_tensor is not None:
        scale = scale_tensor.to(device=hidden_states.device, dtype=torch.float32).reshape(scale_shape)
    elif scale_values is not None:
        # Backward compatibility for early experiments that carried the scale as
        # a Python list in metadata. New transport sends it as a tiny tensor.
        scale = torch.tensor(scale_values, dtype=torch.float32, device=hidden_states.device).reshape(scale_shape)
    else:
        raise ValueError("Missing S2S activation quantization scale tensor")
    while scale.ndim < hidden_states.ndim:
        scale = scale.unsqueeze(-1)
    restored = hidden_states.float() * scale

    dtype_name = str(quant_meta.get("orig_dtype", "float16"))
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name, torch.float16)
    restored = restored.to(dtype=dtype)
    orig_shape = tuple(int(dim) for dim in quant_meta.get("orig_shape", ()))
    if orig_shape and tuple(restored.shape) != orig_shape:
        restored = restored.reshape(orig_shape)

    if logger is not None and os.environ.get(_LOGS_ENV, "0") == "1":
        logger.info(
            "[S2S_ACTIVATION_DEQUANT] context=%s scheme=int8_per_token shape=%s dtype=%s",
            context,
            tuple(restored.shape),
            dtype_name,
        )
    return restored.contiguous()
