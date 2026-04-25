"""Phase 4 sanity tests: K=V on full layers + shared_kv_states={} pass-through.

On Gemma-4-31B the full-attention layers set `attention_k_eq_v=True`,
which causes HF's `Gemma4TextAttention.__init__` to leave `v_proj`
unconstructed. Our block wrapper doesn't need special logic — HF's
attention module itself aliases V=K inside forward. These tests just
make sure the BloomBee side of the contract (the (K, V) tuple returned
from the block) is shape-correct despite the aliasing.

Also spot-check that passing `shared_kv_states={}` through
`super().forward()` doesn't blow up on 31B's config (where
`num_kv_shared_layers=0`).
"""

import pytest
import torch

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from bloombee.models.gemma4.block import WrappedGemma4Block


def _mini_cfg():
    """Tiny Gemma-4-shaped config with both layer types present."""
    cfg = Gemma4TextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        num_global_key_value_heads=2,
        global_head_dim=16,
        attention_k_eq_v=True,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1e6,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
        rms_norm_eps=1e-6,
        vocab_size=256,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        intermediate_size=128,
        final_logit_softcapping=30.0,
        attn_implementation="eager",
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_full_layer_has_no_v_proj_and_uses_alternative_attention():
    """Confirm HF's attention module drops v_proj on the full layer."""
    cfg = _mini_cfg()
    sliding = WrappedGemma4Block(cfg, layer_idx=0).eval()
    full = WrappedGemma4Block(cfg, layer_idx=1).eval()

    assert sliding.self_attn.v_proj is not None, "sliding layer should have v_proj"
    assert sliding.self_attn.use_alternative_attention is False

    assert full.self_attn.v_proj is None, "full layer's v_proj must be None when attention_k_eq_v"
    assert full.self_attn.use_alternative_attention is True


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_block_forward_returns_shape_correct_kv(layer_idx):
    """Both sliding (v_proj present) and full (v_proj None, V aliases K)
    layers must produce usable (K, V) in BloomBee's 3D contract."""
    torch.manual_seed(0)
    cfg = _mini_cfg()
    block = WrappedGemma4Block(cfg, layer_idx=layer_idx).eval()

    seq_len = 4
    h = torch.randn(1, seq_len, cfg.hidden_size)
    out, kv = block(h, use_cache=True)

    assert out.shape == (1, seq_len, cfg.hidden_size)
    pk, pv = kv
    nkv, hd = block._kv_shape()
    assert pk.shape == (nkv, hd, seq_len), f"K shape {pk.shape}, expected {(nkv, hd, seq_len)}"
    assert pv.shape == (nkv, seq_len, hd), f"V shape {pv.shape}, expected {(nkv, seq_len, hd)}"


def test_kv_shape_differs_on_full_vs_sliding():
    """The whole point of Gemma-4 31B: sliding uses num_key_value_heads /
    head_dim; full uses num_global_key_value_heads / global_head_dim. That
    must propagate all the way through the forward to the returned KV tuple."""
    torch.manual_seed(0)
    cfg = _mini_cfg()
    sliding = WrappedGemma4Block(cfg, layer_idx=0).eval()
    full = WrappedGemma4Block(cfg, layer_idx=1).eval()

    h = torch.randn(1, 4, cfg.hidden_size)
    _, (pk_s, pv_s) = sliding(h, use_cache=True)
    _, (pk_f, pv_f) = full(h, use_cache=True)

    # Sliding: 4 KV heads * 8 head_dim → BH=4, D=8.
    assert pk_s.shape == (4, 8, 4)
    assert pv_s.shape == (4, 4, 8)
    # Full: 2 global KV heads * 16 global_head_dim → BH=2, D=16.
    assert pk_f.shape == (2, 16, 4)
    assert pv_f.shape == (2, 4, 16)

    # And they must not accidentally be equal
    assert pk_s.shape != pk_f.shape


def test_shared_kv_states_empty_dict_does_not_crash():
    """We pass shared_kv_states={} unconditionally. On 31B with
    num_kv_shared_layers=0 no layer reads or writes to it, so we just
    need to confirm the API accepts the empty dict without error."""
    torch.manual_seed(0)
    cfg = _mini_cfg()
    assert cfg.num_kv_shared_layers == 0  # sanity for the test's premise

    for layer_idx in (0, 1):
        block = WrappedGemma4Block(cfg, layer_idx=layer_idx).eval()
        h = torch.randn(1, 3, cfg.hidden_size)
        out, _ = block(h, use_cache=True)  # forward internally passes shared_kv_states={}
        assert out.shape == (1, 3, cfg.hidden_size)


def test_prefill_then_decode_round_trip_on_full_layer():
    """K=V aliasing on the full layer shouldn't break the prefill → decode
    BloomBee cache contract: we still read back only the NEW tokens in KV."""
    torch.manual_seed(0)
    cfg = _mini_cfg()
    block = WrappedGemma4Block(cfg, layer_idx=1).eval()  # full layer

    # Prefill
    h0 = torch.randn(1, 5, cfg.hidden_size)
    _, (pk0, pv0) = block(h0, use_cache=True)
    assert pk0.shape[-1] == 5

    # Decode step
    h1 = torch.randn(1, 1, cfg.hidden_size)
    out1, (pk1, pv1) = block(h1, layer_past=(pk0, pv0), use_cache=True)
    assert out1.shape == (1, 1, cfg.hidden_size)
    # Only the NEW 1 token is returned in KV (BloomBee's cumulative cache
    # is managed externally by MemoryCache)
    assert pk1.shape[-1] == 1
    assert pv1.shape[-2] == 1
