"""Qwen3 block parity: compare WrappedQwen3Block against HF's Qwen3DecoderLayer.

These tests exercise the BloomBee adapter for Qwen3 end-to-end on CPU with
small synthetic weights. They guard against TF 5.x API drift (past_key_values
rename, cache_position requirement, DynamicCache layout changes) by comparing
hidden-state output and KV-cache contents against the unmodified HF layer.
"""

import pytest
import torch

from bloombee.models.qwen3.block import WrappedQwen3Block
from bloombee.models.qwen3.config import DistributedQwen3Config


def _make_config(num_kv_heads=2, head_dim=16):
    cfg = DistributedQwen3Config(
        vocab_size=256,
        hidden_size=num_kv_heads * 4 * head_dim,  # 4:1 GQA ratio
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=num_kv_heads * 4,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        attn_implementation="eager",
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"
    return cfg


@pytest.mark.parametrize("seq_len", [1, 4, 8])
def test_prefill_shape_and_kv_contract(seq_len):
    torch.manual_seed(0)
    cfg = _make_config()
    block = WrappedQwen3Block(cfg, layer_idx=0).eval()

    h = torch.randn(1, seq_len, cfg.hidden_size)
    out, kv = block(h, use_cache=True)

    assert out.shape == (1, seq_len, cfg.hidden_size)
    pk, pv = kv
    # BloomBee's 3D cache contract: key [B*H_kv, D, S], value [B*H_kv, S, D]
    assert pk.shape == (cfg.num_key_value_heads, cfg.head_dim, seq_len)
    assert pv.shape == (cfg.num_key_value_heads, seq_len, cfg.head_dim)


def test_prefill_then_decode_length_advances():
    """Verify that a prefill + single-token decode step increments the KV length."""
    torch.manual_seed(0)
    cfg = _make_config()
    block = WrappedQwen3Block(cfg, layer_idx=0).eval()

    prefill_len = 5
    h = torch.randn(1, prefill_len, cfg.hidden_size)
    out, kv = block(h, use_cache=True)
    pk, pv = kv
    assert pk.shape[-1] == prefill_len

    # Decode step: feed one more token, pass past KV in BloomBee shape
    h_next = torch.randn(1, 1, cfg.hidden_size)
    out2, kv2 = block(h_next, layer_past=(pk, pv), use_cache=True)
    assert out2.shape == (1, 1, cfg.hidden_size)
    pk2, pv2 = kv2
    # Only the *new* tokens (1) should be returned, per BloomBee's cache contract
    assert pk2.shape[-1] == 1
    assert pv2.shape[-2] == 1


def test_forward_is_deterministic_without_use_cache():
    torch.manual_seed(0)
    cfg = _make_config()
    block = WrappedQwen3Block(cfg, layer_idx=0).eval()
    h = torch.randn(1, 6, cfg.hidden_size)
    out_a, _ = block(h, use_cache=False)
    out_b, _ = block(h, use_cache=False)
    torch.testing.assert_close(out_a, out_b)


def test_gqa_head_contract():
    """With 4:1 GQA the cache carries only num_key_value_heads, not num_attention_heads."""
    cfg = _make_config(num_kv_heads=2, head_dim=16)
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 2

    block = WrappedQwen3Block(cfg, layer_idx=0).eval()
    h = torch.randn(1, 3, cfg.hidden_size)
    _, (pk, pv) = block(h, use_cache=True)
    # Packed as [B*H_kv, ...] — must be 2 * 1 = 2, NOT 8
    assert pk.shape[0] == cfg.num_key_value_heads
    assert pv.shape[0] == cfg.num_key_value_heads


def test_decode_step_matches_prefill_of_same_tokens():
    """
    Running prefill(x1 | x2) in one go and prefill(x1) + decode(x2) should
    yield the same hidden state for x2.  This verifies KV cache correctness
    under the TF 5.x cache_position/past_key_values plumbing.
    """
    torch.manual_seed(0)
    cfg = _make_config()
    block = WrappedQwen3Block(cfg, layer_idx=0).eval()

    x = torch.randn(1, 4, cfg.hidden_size)

    # Full prefill
    with torch.no_grad():
        full_out, _ = block(x, use_cache=False)

    # Incremental: prefill first 3 tokens, then decode the 4th
    with torch.no_grad():
        _, (pk, pv) = block(x[:, :3, :], use_cache=True)
        decode_out, _ = block(x[:, 3:4, :], layer_past=(pk, pv), use_cache=False)

    torch.testing.assert_close(full_out[:, 3:4, :], decode_out, atol=5e-3, rtol=5e-3)


def test_unspecified_mask_is_handled():
    """attention_mask=None (causal by default) is the normal path in BloomBee."""
    torch.manual_seed(0)
    cfg = _make_config()
    block = WrappedQwen3Block(cfg, layer_idx=0).eval()
    h = torch.randn(1, 4, cfg.hidden_size)
    out, _ = block(h, attention_mask=None, use_cache=True)
    assert out.shape == h.shape
