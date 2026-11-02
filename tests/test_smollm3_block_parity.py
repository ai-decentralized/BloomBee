"""SmolLM3 block parity: compare WrappedSmolLM3Block against HF's SmolLM3 on CPU.

SmolLM3 is Llama-like with GQA, but every 4th layer is a NoPE layer (no rotary
embedding) and configs may enable sliding-window attention. These tests chain
wrapped blocks with synthetic weights and compare them against the unmodified
HF SmolLM3Model, both for a single prefill and for prefill + token-by-token
decode through BloomBee's KV cache contract.
"""

import pytest
import torch

pytest.importorskip("transformers.models.smollm3")

from transformers.models.smollm3 import SmolLM3Model

from bloombee.models.smollm3.block import WrappedSmolLM3Block
from bloombee.models.smollm3.config import DistributedSmolLM3Config


def _make_config(num_layers=4, num_kv_heads=2, head_dim=16, **kwargs):
    cfg = DistributedSmolLM3Config(
        vocab_size=256,
        hidden_size=num_kv_heads * 4 * head_dim,  # 4:1 GQA ratio
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=num_kv_heads * 4,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _head_dim(cfg):
    # SmolLM3Config has no head_dim field; attention derives it from hidden_size.
    return cfg.hidden_size // cfg.num_attention_heads


def _hf_model_and_blocks(cfg):
    torch.manual_seed(0)
    hf = SmolLM3Model(cfg).eval()
    blocks = []
    for i, layer in enumerate(hf.layers):
        block = WrappedSmolLM3Block(cfg, layer_idx=i).eval()
        missing, unexpected = block.load_state_dict(layer.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("_rotary_emb.") for k in missing)
        blocks.append(block)
    return hf, blocks


def _run_blocks(blocks, hidden, caches=None):
    """Run hidden states through the chain; caches is a per-block list of BloomBee (k, v) or None."""
    new_caches = []
    for i, block in enumerate(blocks):
        past = None if caches is None else caches[i]
        hidden, kv = block(hidden, layer_past=past, use_cache=True)
        new_caches.append(kv)
    return hidden, new_caches


def _append(past, new):
    if past is None:
        return new
    # BloomBee 3D layout: key [B*H_kv, D, S], value [B*H_kv, S, D]
    return (torch.cat([past[0], new[0]], dim=2), torch.cat([past[1], new[1]], dim=1))


def test_default_config_has_nope_layers():
    cfg = _make_config(num_layers=8)
    assert cfg.no_rope_layers == [1, 1, 1, 0, 1, 1, 1, 0]
    assert all(lt == "full_attention" for lt in cfg.layer_types)
    block = WrappedSmolLM3Block(cfg, layer_idx=3)
    assert not block.self_attn.use_rope
    assert block.sliding_window is None


@pytest.mark.parametrize("seq_len", [1, 5])
def test_kv_contract(seq_len):
    cfg = _make_config()
    block = WrappedSmolLM3Block(cfg, layer_idx=0).eval()
    out, (pk, pv) = block(torch.randn(2, seq_len, cfg.hidden_size), use_cache=True)
    assert out.shape == (2, seq_len, cfg.hidden_size)
    # Packed as [B*H_kv, ...]: only num_key_value_heads, not num_attention_heads
    assert pk.shape == (2 * cfg.num_key_value_heads, _head_dim(cfg), seq_len)
    assert pv.shape == (2 * cfg.num_key_value_heads, seq_len, _head_dim(cfg))


def test_rotary_inv_freq_stays_fp32_under_fp16_cast():
    block = WrappedSmolLM3Block(_make_config(), layer_idx=0).eval()
    block.to(torch.float16)
    assert block._rotary_emb.inv_freq.dtype == torch.float32


@pytest.mark.parametrize("sliding", [False, True])
def test_prefill_matches_hf_model(sliding):
    """Chained wrapped blocks (including a NoPE layer) == HF SmolLM3Model on a full prefill."""
    extra = dict(use_sliding_window=True, sliding_window=3) if sliding else {}
    cfg = _make_config(**extra)
    if sliding:
        assert "sliding_attention" in cfg.layer_types
    hf, blocks = _hf_model_and_blocks(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        ref = hf(input_ids=input_ids).last_hidden_state
        hidden, _ = _run_blocks(blocks, hf.embed_tokens(input_ids))
        ours = hf.norm(hidden)

    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("sliding", [False, True])
def test_incremental_decode_matches_hf_model(sliding):
    """Prefill 4 tokens, then decode one token at a time through BloomBee-layout caches."""
    extra = dict(use_sliding_window=True, sliding_window=3) if sliding else {}
    cfg = _make_config(**extra)
    hf, blocks = _hf_model_and_blocks(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        ref = hf(input_ids=input_ids).last_hidden_state
        embeds = hf.embed_tokens(input_ids)

        hidden, caches = _run_blocks(blocks, embeds[:, :4])
        outs = [hidden]
        for t in range(4, 8):
            hidden, new = _run_blocks(blocks, embeds[:, t : t + 1], caches)
            caches = [_append(past, kv) for past, kv in zip(caches, new)]
            outs.append(hidden)
        ours = hf.norm(torch.cat(outs, dim=1))

    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_4d_cache_with_padded_heads():
    """Backend may hand back a 4D [B, H_attn, S, D] cache; only the first H_kv heads are valid."""
    cfg = _make_config()
    block = WrappedSmolLM3Block(cfg, layer_idx=0).eval()
    x = torch.randn(1, 5, cfg.hidden_size)
    with torch.no_grad():
        full, _ = block(x, use_cache=False)
        _, (pk, pv) = block(x[:, :4], use_cache=True)
        nkv, d = cfg.num_key_value_heads, _head_dim(cfg)
        k4 = torch.zeros(1, cfg.num_attention_heads, 4, d)
        v4 = torch.zeros(1, cfg.num_attention_heads, 4, d)
        k4[:, :nkv] = pk.permute(0, 2, 1).reshape(1, nkv, 4, d)
        v4[:, :nkv] = pv.reshape(1, nkv, 4, d)
        step, _ = block(x[:, 4:5], layer_past=(k4, v4), use_cache=True)
    torch.testing.assert_close(step, full[:, 4:5], atol=1e-4, rtol=1e-4)


def test_backend_head_dim_without_config_field():
    """Regression: SmolLM3 configs expose per_layer_config but no head_dim field.
    TransformerBackend used to read per_layer_config[i].head_dim and crash at server start."""
    from bloombee.server.backend import TransformerBackend

    cfg = _make_config()
    assert cfg.per_layer_config and getattr(cfg.per_layer_config[0], "head_dim", None) is None
    for block_index in range(cfg.num_hidden_layers):
        backend = TransformerBackend.__new__(TransformerBackend)  # skip ModuleBackend init
        backend.config = cfg
        backend.block_index = block_index
        assert backend._head_dim_for_this_block() == _head_dim(cfg)
