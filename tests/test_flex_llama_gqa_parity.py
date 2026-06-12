"""Numerical parity for the FlexGen llama GQA path.

mha_llama derives the kv head count from w_k's rows and broadcasts k/v to
the attention heads after rotary. Verify against a plain PyTorch reference
(same weights, same rotary helpers) and check the returned cache layout."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.forked
def test_mha_llama_gqa_matches_reference():
    from bloombee.flexgen_utils.pytorch_backend import (
        TorchDevice,
        TorchTensor,
        apply_rotary_emb,
        precompute_freqs_cis,
        rms_norm,
    )

    device = TorchDevice("cpu")
    torch.manual_seed(7)

    batch, seq = 2, 6
    n_head, n_kv, head_dim = 8, 2, 16
    hidden = n_head * head_dim  # 128
    kv_dim = n_kv * head_dim    # 32
    dtype = torch.float32       # fp32 keeps the comparison tight on CPU

    w_q = torch.randn(hidden, hidden, dtype=dtype) * 0.05
    w_k = torch.randn(kv_dim, hidden, dtype=dtype) * 0.05
    w_v = torch.randn(kv_dim, hidden, dtype=dtype) * 0.05
    w_o = torch.randn(hidden, hidden, dtype=dtype) * 0.05
    ln = torch.rand(hidden, dtype=dtype) + 0.5
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    x = torch.randn(batch, seq, hidden, dtype=dtype) * 0.5

    causal = torch.zeros(batch, 1, seq, seq, dtype=dtype)
    causal.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), float("-inf"))

    def tt(t):
        return TorchTensor.create_from_torch(t, device)

    out, new_k, new_v = device.mha_llama(
        tt(x.clone()), tt(causal), tt(w_q), tt(w_k), tt(w_v), tt(w_o),
        n_head, (False, False), False, None, tt(ln), tt(inv_freq), None,
    )

    # ---- plain PyTorch reference (same rotary helpers) ----
    h = rms_norm(x, ln)
    q = F.linear(h, w_q).view(batch, seq, n_head, head_dim)
    k = F.linear(h, w_k).view(batch, seq, n_kv, head_dim)
    v = F.linear(h, w_v).view(batch, seq, n_kv, head_dim)
    freqs = precompute_freqs_cis(head_dim, 2048 * 2, inv_freq, position_ids=None)[:seq]
    q, k = apply_rotary_emb(q, k, freqs_cis=freqs)
    groups = n_head // n_kv
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    q = q.permute(0, 2, 1, 3)            # (b, n_head, s, d)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = q @ k.transpose(-1, -2) / (head_dim ** 0.5) + causal
    probs = F.softmax(scores.float(), dim=-1).to(dtype)
    ctx = (probs @ v).transpose(1, 2).reshape(batch, seq, hidden)
    ref = F.linear(ctx, w_o) + x

    torch.testing.assert_close(out.data, ref, rtol=2e-4, atol=2e-4)

    # ---- cache layout: (s, b*n_head, d) with kv groups identical ----
    assert new_k.data.shape[-2:] == (batch * n_head, head_dim) or new_k.data.shape[0] == batch * n_head
    k_cache = new_k.data
    if k_cache.shape[0] != seq:  # key cache is (b*n_head, d, s); normalize to (s, b*n_head, d)
        k_cache = k_cache.permute(2, 0, 1)
    per_head = k_cache.view(seq, batch, n_head, head_dim)
    for g in range(n_kv):
        block = per_head[:, :, g * groups:(g + 1) * groups, :]
        torch.testing.assert_close(block, block[:, :, :1, :].expand_as(block))
