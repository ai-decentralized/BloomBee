"""End-to-end decode parity for FlexGen mha_gen_llama across the two
cache-write branches.

Phase 0 changed the write semantics inside ``mha_gen_llama``: when the
cache capacity >= src_s we now write k_new/v_new into the preallocated
slab in place; otherwise we fall back to ``torch.cat``. The branches
should be functionally equivalent.

This test builds a minimally-plumbed ``mha_gen_llama`` call (real layer
weights, real rotary, real attention kernel) twice:

1. Preallocated slab (new path, cache_capacity > src_s).
2. History-shaped cache (legacy path, cache_capacity == history_len).

Both receive the same inputs and mask; the attention-weighted value
output must match to fp16 rounding.

Gated on BLOOMBEE_DECODE_PARITY=1 because it requires CUDA and pulls in
the full pytorch_backend module. Runs in ~2 s on V100.
"""

from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("BLOOMBEE_DECODE_PARITY") != "1",
    reason="set BLOOMBEE_DECODE_PARITY=1 (CUDA-only test)",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mha_gen_llama_branches_agree_on_identical_inputs():
    from bloombee.flexgen_utils.pytorch_backend import (
        TorchDevice,
        TorchTensor,
        DeviceType,
    )

    device = TorchDevice("cuda:0")

    # llama-7b-ish decode shapes, scaled to fit the test box.
    batch = 1
    n_head = 8
    head_dim = 64
    hidden_size = n_head * head_dim  # 512
    history_len = 31
    tgt_s = 1
    src_s = history_len + tgt_s
    dtype = torch.float16
    dev = "cuda:0"

    torch.manual_seed(0)

    # Layer weights (row-major, F.linear-style)
    w_q = torch.randn(hidden_size, hidden_size, dtype=dtype, device=dev) * 0.02
    w_k = torch.randn(hidden_size, hidden_size, dtype=dtype, device=dev) * 0.02
    w_v = torch.randn(hidden_size, hidden_size, dtype=dtype, device=dev) * 0.02
    w_out = torch.randn(hidden_size, hidden_size, dtype=dtype, device=dev) * 0.02
    input_layernorm = torch.ones(hidden_size, dtype=dtype, device=dev)

    # Llama rotary freq table; pre-computed inverse frequencies.
    rotary_inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim)
    )

    hidden_states = torch.randn(batch, tgt_s, hidden_size, dtype=dtype, device=dev) * 0.5

    # KV history at FlexGen layout (S, B*H, D).
    BH = batch * n_head
    history_k = torch.randn(history_len, BH, head_dim, dtype=dtype, device=dev)
    history_v = torch.randn(history_len, BH, head_dim, dtype=dtype, device=dev)

    # Build an attention mask shaped (1, 1, src_s) of the same data that
    # FlexGen feeds: a causal prefix (all ones) for decode.
    attn_mask = torch.ones(batch, tgt_s, src_s, dtype=torch.bool, device=dev)

    # ---- Path A: preallocated slab (cache_capacity = 2 * src_s > src_s) ----
    slab_cap = src_s * 2
    slab_k = torch.zeros(slab_cap, BH, head_dim, dtype=dtype, device=dev)
    slab_v = torch.zeros(slab_cap, BH, head_dim, dtype=dtype, device=dev)
    slab_k[:history_len] = history_k
    slab_v[:history_len] = history_v

    # ---- Path B: history-shaped (cache_capacity == history_len) — concat path ----
    tight_k = history_k.clone()
    tight_v = history_v.clone()

    def _run(k_cache_tensor, v_cache_tensor):
        w_q_tt = TorchTensor.create_from_torch(w_q, device)
        w_k_tt = TorchTensor.create_from_torch(w_k, device)
        w_v_tt = TorchTensor.create_from_torch(w_v, device)
        w_out_tt = TorchTensor.create_from_torch(w_out, device)
        ln_tt = TorchTensor.create_from_torch(input_layernorm, device)
        rotary_tt = TorchTensor.create_from_torch(rotary_inv_freq, device)
        mask_tt = TorchTensor.create_from_torch(attn_mask, device)
        hs_tt = TorchTensor.create_from_torch(hidden_states.clone(), device)

        k_cache_tt = TorchTensor.create_from_torch(k_cache_tensor, device)
        v_cache_tt = TorchTensor.create_from_torch(v_cache_tensor, device)

        out, _, _ = device.mha_gen_llama(
            inputs=hs_tt,
            attention_mask=mask_tt,
            w_q=w_q_tt,
            w_k=w_k_tt,
            w_v=w_v_tt,
            w_out=w_out_tt,
            n_head=n_head,
            k_cache=k_cache_tt,
            v_cache=v_cache_tt,
            donate=[False] * 11,
            attn_sparsity=1.0,
            compress_cache=False,
            comp_config=None,
            input_layernorm=ln_tt,
            rotary_emb_inv_freq=rotary_tt,
            rotary_position_ids=None,
        )
        return out.data

    out_a = _run(slab_k, slab_v)
    out_b = _run(tight_k, tight_v)

    # Must match to fp16 rounding; both take identical math, just different
    # storage layouts for the K/V prefix.
    torch.testing.assert_close(out_a, out_b, atol=1e-3, rtol=1e-3)
