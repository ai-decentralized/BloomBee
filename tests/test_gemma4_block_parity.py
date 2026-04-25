"""Gemma-4 block wrapper: minimal shape/contract test.

Covers the skeleton of `WrappedGemma4Block`:
  - builds on a tiny synthetic text-only Gemma4TextConfig;
  - runs one prefill + one decode step on each of the two layer types
    (sliding_attention, full_attention);
  - checks that BloomBee's 3D KV cache contract is honored.

Does NOT exercise the checkpoint — no weight loading, no HF tokenizer.
"""

import pytest
import torch

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from bloombee.models.gemma4.block import WrappedGemma4Block


def _make_mini_config(layer_type: str = "sliding_attention"):
    """Smallest Gemma-4-shaped config that fits on CPU in seconds.

    We keep the proportions of the 31B model roughly intact (tight GQA
    on full-attn layers, wider KV heads on sliding layers, dual head_dim)
    so the block wrapper exercises the same branching it will hit on A100.
    """
    # Two layers: one of each type so we can test both branches.
    cfg = Gemma4TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,          # sliding path: 8 q / 4 kv, head_dim=16
        head_dim=16,
        num_global_key_value_heads=2,   # full path: 8 q / 2 kv, head_dim=32
        global_head_dim=32,
        attention_k_eq_v=True,
        sliding_window=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=30.0,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    cfg._attn_implementation = "eager"
    return cfg


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_prefill_produces_valid_shapes_and_kv(layer_idx):
    """Both sliding (idx 0) and full (idx 1) layers must run cleanly and
    return BloomBee's 3D KV contract."""
    torch.manual_seed(0)
    cfg = _make_mini_config()
    block = WrappedGemma4Block(cfg, layer_idx=layer_idx).eval()

    seq_len = 6
    h = torch.randn(1, seq_len, cfg.hidden_size)
    out, kv = block(h, use_cache=True)

    assert out.shape == (1, seq_len, cfg.hidden_size)

    pk, pv = kv
    # BloomBee 3D cache contract: key [B*H_kv, D, S], value [B*H_kv, S, D]
    nkv, hd = block._kv_shape()
    assert pk.shape == (nkv, hd, seq_len)
    assert pv.shape == (nkv, seq_len, hd)


def test_full_attention_layer_uses_different_kv_shape_than_sliding():
    """The heart of the Gemma-4 quirk: sliding vs full use different
    (num_kv_heads, head_dim) pairs. We must see it in the cache shapes."""
    torch.manual_seed(0)
    cfg = _make_mini_config()
    sliding = WrappedGemma4Block(cfg, layer_idx=0).eval()  # sliding
    full = WrappedGemma4Block(cfg, layer_idx=1).eval()     # full

    assert sliding._kv_shape() != full._kv_shape(), (
        f"sliding kv_shape={sliding._kv_shape()} == full kv_shape={full._kv_shape()} "
        "— the whole point of Gemma-4's dual-head-dim layout"
    )
    assert sliding._kv_shape() == (cfg.num_key_value_heads, cfg.head_dim)
    assert full._kv_shape() == (cfg.num_global_key_value_heads, cfg.global_head_dim)


def test_decode_step_after_prefill_extends_kv():
    """Feed a 1-token follow-up after prefill; the cache contract must
    only return the *new* 1 token's KV, not the full prefix."""
    torch.manual_seed(0)
    cfg = _make_mini_config()
    block = WrappedGemma4Block(cfg, layer_idx=0).eval()

    prefill_len = 5
    h0 = torch.randn(1, prefill_len, cfg.hidden_size)
    _, kv0 = block(h0, use_cache=True)
    pk0, pv0 = kv0
    assert pk0.shape[-1] == prefill_len

    h1 = torch.randn(1, 1, cfg.hidden_size)
    out1, kv1 = block(h1, layer_past=(pk0, pv0), use_cache=True)
    assert out1.shape == (1, 1, cfg.hidden_size)
    pk1, pv1 = kv1
    # BloomBee cache contract: present KV covers only the *new* tokens,
    # the cumulative concatenation is managed externally by MemoryCache.
    assert pk1.shape[-1] == 1
    assert pv1.shape[-2] == 1


def test_rotary_buffers_stay_fp32_after_half_cast():
    """Regression: Gemma-4's rotary holds one inv_freq pair per layer type;
    _apply must keep all of them in fp32 across .half() so rotary positions
    don't accumulate quantization error."""
    torch.manual_seed(0)
    cfg = _make_mini_config()
    block = WrappedGemma4Block(cfg, layer_idx=0)
    block = block.half()

    rot = block._rotary_emb
    for lt in rot.layer_types:
        for suffix in ("inv_freq", "original_inv_freq"):
            buf = getattr(rot, f"{lt}_{suffix}")
            assert buf.dtype == torch.float32, (
                f"{lt}_{suffix} got {buf.dtype}, expected float32"
            )
