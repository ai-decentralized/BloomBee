"""Opt-in lossy codecs for activations / activation gradients crossing the network in LoRA training.

  none           exact (default)
  int8           absmax int8 per (token, block of 32 channels) + bf16 scale: ~1.9x fewer bytes
                 than fp16. Small blocks matter: LLM hidden states contain "massive
                 activation" channels 100-1000x larger than the rest, and with one scale per
                 whole token they would crush every other channel to zero.
  hadamard-int8  the same after an orthonormal Hadamard rotation inside each block of 32,
                 which spreads outliers within the block (as in QuaRot / TAH-Quant).

Only boundary activations and their gradients use a codec; adapters and adapter gradients
always travel as exact fp32. Encoding is deterministic, so a stateless retry sees exactly
the values the cached path saw.
"""

import math

import torch

CODECS = ("none", "int8", "hadamard-int8")
_BLOCK = 32
_HADAMARD = {}


def _hadamard(n, device):
    key = (n, device)
    if key not in _HADAMARD:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _HADAMARD[key] = (h / math.sqrt(n)).to(device)
    return _HADAMARD[key]


def encode(tensor, codec):
    """Return the list of tensors to put on the wire."""
    if codec == "none":
        return [tensor]
    if codec not in CODECS:
        raise ValueError(f"Unknown LoRA boundary codec {codec!r}; choose from {CODECS}")
    x = tensor.float()
    width = x.shape[-1]
    if width % _BLOCK == 0:
        x = x.reshape(*x.shape[:-1], width // _BLOCK, _BLOCK)
        if codec == "hadamard-int8":
            x = x @ _hadamard(_BLOCK, x.device)
    else:
        x = x.unsqueeze(-2)  # width not a multiple of the block: one block per token
    # bf16 scales (fp32 exponent range): activation *gradients* are ~1e-6 and would underflow fp16.
    scale = (x.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-30).to(torch.bfloat16)
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
    if codec == "hadamard-int8" and width // blocks == _BLOCK:
        x = x @ _hadamard(_BLOCK, x.device).T
    return x.reshape(q.shape).to(dtype)


def wire_count(codec):
    return 1 if codec == "none" else 2
