"""Opt-in lossy codecs for activations / activation gradients crossing the network in LoRA training.

  none           exact (default)
  int8           per-token absmax int8 + fp16 scale: ~2x fewer bytes than fp16
  hadamard-int8  orthonormal block-Hadamard rotation (blocks of 128 along hidden), then
                 per-(token, block) absmax int8. Rotation spreads the few huge outlier
                 channels typical of LLM hidden states over the block, so int8 keeps far
                 more precision (as in QuaRot / TAH-Quant).

Only boundary activations and their gradients use a codec; adapters and adapter gradients
always travel as exact fp32. Encoding is deterministic, so a stateless retry sees exactly
the values the cached path saw.
"""

import math

import torch

CODECS = ("none", "int8", "hadamard-int8")
_BLOCK = 128
_HADAMARD = {}


def _hadamard(n, device):
    key = (n, device)
    if key not in _HADAMARD:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _HADAMARD[key] = (h / math.sqrt(n)).to(device)
    return _HADAMARD[key]


def _uses_hadamard(codec, width):
    return codec == "hadamard-int8" and width % _BLOCK == 0


def encode(tensor, codec):
    """Return the list of tensors to put on the wire."""
    if codec == "none":
        return [tensor]
    if codec not in CODECS:
        raise ValueError(f"Unknown LoRA boundary codec {codec!r}; choose from {CODECS}")
    x = tensor.float()
    width = x.shape[-1]
    if _uses_hadamard(codec, width):
        x = (x.reshape(-1, _BLOCK) @ _hadamard(_BLOCK, x.device)).reshape(*x.shape[:-1], width // _BLOCK, _BLOCK)
    else:
        x = x.unsqueeze(-2)  # one block per token
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    scale = scale.to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
    q = torch.round(x / scale.float()).clamp_(-127, 127).to(torch.int8)
    return [q.reshape(tensor.shape), scale.squeeze(-1)]


def decode(parts, codec, dtype):
    if codec == "none":
        (tensor,) = parts
        return tensor
    q, scale = parts
    width = q.shape[-1]
    blocks = scale.shape[-1]
    x = q.float().reshape(*q.shape[:-1], blocks, width // blocks) * scale.float().unsqueeze(-1)
    if _uses_hadamard(codec, width):
        x = (x.reshape(-1, _BLOCK) @ _hadamard(_BLOCK, x.device).T).reshape(q.shape)
    return x.reshape(q.shape).to(dtype)


def wire_count(codec):
    return 1 if codec == "none" else 2
