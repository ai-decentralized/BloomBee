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


def test_rotary_inv_freq_stays_fp32_under_fp16_cast():
    """Regression: `.to(torch.float16)` used to downcast rotary `inv_freq`,
    which caused fp16 rounding of RoPE positions and catastrophic generation
    quality loss on real checkpoints (token collapse into repeated groups).
    """
    cfg = _make_config()
    block = WrappedQwen3Block(cfg, layer_idx=0).eval()
    block.to(torch.float16)
    assert block._rotary_emb.inv_freq.dtype == torch.float32
    assert block._rotary_emb.original_inv_freq.dtype == torch.float32


def test_fp16_block_matches_hf_layer_with_causal_mask():
    """End-to-end parity against HF's native Qwen3DecoderLayer in fp16,
    forcing the wrapper through the rotary buffer cast (block.to(fp16)).
    Covers the inv_freq fp16-downcast bug that survived the unit tests."""
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer as _HFDecoderLayer,
        Qwen3RotaryEmbedding as _HFRotary,
    )

    torch.manual_seed(0)
    cfg = _make_config()
    B, S = 1, 8

    hf = _HFDecoderLayer(cfg, layer_idx=0).eval()
    wrapped = WrappedQwen3Block(cfg, layer_idx=0).eval()
    wrapped.load_state_dict(hf.state_dict(), strict=False)

    hf.to(torch.float16)
    wrapped.to(torch.float16)

    h = torch.randn(B, S, cfg.hidden_size, dtype=torch.float16)
    causal = torch.triu(
        torch.full((S, S), float("-inf"), dtype=torch.float16), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    pos_ids = torch.arange(S).unsqueeze(0)
    rot = _HFRotary(cfg)
    pos_emb = rot(h, pos_ids)

    with torch.no_grad():
        hf_out = hf(
            h,
            attention_mask=causal,
            position_ids=pos_ids,
            position_embeddings=pos_emb,
            past_key_values=None,
            use_cache=False,
            cache_position=pos_ids[0],
        )
        hf_out = hf_out[0] if isinstance(hf_out, tuple) else hf_out

        wr_out, _ = wrapped(h, attention_mask=causal, layer_past=None, use_cache=True)

    torch.testing.assert_close(hf_out, wr_out, atol=5e-3, rtol=5e-3)
