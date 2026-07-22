"""DeepSeek-V3 block parity: compare WrappedDeepseekV3Block against HF's DeepseekV3DecoderLayer.

These tests exercise the BloomBee adapter for DeepSeek-V3 end-to-end on CPU with
small synthetic weights. DeepSeek-V3 uses Multi-head Latent Attention (MLA), where
the per-head key width (qk_head_dim) and value width (v_head_dim) differ, plus a
mixture of dense and MoE decoder layers (first_k_dense_replace). This guards the
cache padding/stripping in WrappedDeepseekV3Block._reorder_cache_{from,to}_bloom
and exercises both the dense and MoE layer variants.
"""

import pytest
import torch

from bloombee.models.deepseekv3.block import WrappedDeepseekV3Block
from bloombee.models.deepseekv3.config import DistributedDeepseekV3Config


def _make_config():
    cfg = DistributedDeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        v_head_dim=8,
        qk_nope_head_dim=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=64,
        attn_implementation="eager",
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _new_block(cfg, layer_idx):
    """Construct a block and reinitialize every parameter deterministically.

    DeepseekV3Experts (used by MoE layers) allocates gate_up_proj/down_proj via
    torch.empty(...) rather than a standard init; real usage always overwrites
    these via from_pretrained() or post_init(), which this lightweight
    construction path skips. Uninitialized memory isn't reliably NaN/inf -- it
    can be finite-but-enormous garbage that only shows up as inf after it's
    been multiplied through a few layers, so every parameter must be
    unconditionally reinitialized rather than patched only when already
    non-finite.
    """
    block = WrappedDeepseekV3Block(cfg, layer_idx=layer_idx)
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return block.eval()


@pytest.mark.parametrize("layer_idx", [0, 1])  # 0 = dense MLP layer, 1 = MoE layer
@pytest.mark.parametrize("seq_len", [1, 4, 8])
def test_prefill_shape_and_kv_contract(layer_idx, seq_len):
    torch.manual_seed(0)
    cfg = _make_config()
    block = _new_block(cfg, layer_idx)

    h = torch.randn(1, seq_len, cfg.hidden_size)
    out, kv = block(h, use_cache=True)

    assert out.shape == (1, seq_len, cfg.hidden_size)
    assert torch.isfinite(out).all()

    pk, pv = kv
    qk_d = block.self_attn.qk_head_dim
    # BloomBee's 3D cache contract: key [B*H, qk_head_dim, S], value [B*H, S, qk_head_dim] (zero-padded)
    assert pk.shape == (cfg.num_attention_heads, qk_d, seq_len)
    assert pv.shape == (cfg.num_attention_heads, seq_len, qk_d)


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_prefill_then_decode_length_advances(layer_idx):
    torch.manual_seed(0)
    cfg = _make_config()
    block = _new_block(cfg, layer_idx)

    prefill_len = 5
    h = torch.randn(1, prefill_len, cfg.hidden_size)
    out, kv = block(h, use_cache=True)
    pk, pv = kv
    assert pk.shape[-1] == prefill_len

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
    block = _new_block(cfg, layer_idx=1)
    h = torch.randn(1, 6, cfg.hidden_size)
    out_a, _ = block(h, use_cache=False)
    out_b, _ = block(h, use_cache=False)
    torch.testing.assert_close(out_a, out_b)


def test_matches_reference_decoder_layer_prefill():
    """WrappedDeepseekV3Block's output should match a bare HF DeepseekV3DecoderLayer
    given the same weights and inputs (sanity check on the MLA cache reshape math).
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer

    torch.manual_seed(0)
    cfg = _make_config()

    wrapped = _new_block(cfg, layer_idx=1)
    reference = DeepseekV3DecoderLayer(cfg, layer_idx=1).eval()
    reference.load_state_dict(wrapped.state_dict())

    seq_len = 4
    h = torch.randn(1, seq_len, cfg.hidden_size)

    out_wrapped, _ = wrapped(h, use_cache=False)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    position_embeddings = wrapped._rotary_emb(h, position_ids)
    neg_inf = torch.finfo(h.dtype).min
    causal = torch.triu(torch.full((seq_len, seq_len), neg_inf), diagonal=1)
    attention_mask = causal.unsqueeze(0).unsqueeze(0)

    ref_out = reference(
        h,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        use_cache=False,
    )
    ref_hidden = ref_out[0] if isinstance(ref_out, tuple) else ref_out

    torch.testing.assert_close(out_wrapped, ref_hidden)
