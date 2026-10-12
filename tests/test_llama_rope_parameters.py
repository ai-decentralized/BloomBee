from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from bloombee.models.llama.flex_llama import compute_llama_inv_freq


@pytest.mark.parametrize("rope_type,factor", [("default", None), ("linear", 8.0)])
@pytest.mark.parametrize("theta", [10000.0, 100000.0, 500000.0])
def test_rope_frequencies_match_hf_config(rope_type, factor, theta):
    parameters = {"rope_type": rope_type, "rope_theta": theta}
    if factor is not None:
        parameters["factor"] = factor
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_parameters=parameters,
    )
    expected = LlamaRotaryEmbedding(config).inv_freq
    torch.testing.assert_close(compute_llama_inv_freq(config, 32), expected, rtol=0, atol=0)


def test_modern_rope_parameters_survive_initializer_fallback(monkeypatch):
    from transformers import modeling_rope_utils

    monkeypatch.setattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", {})
    config = SimpleNamespace(rope_parameters={"rope_type": "linear", "rope_theta": 100000.0, "factor": 8.0})
    expected = 1.0 / (100000.0 ** (torch.arange(0, 32, 2).float() / 32)) / 8.0
    torch.testing.assert_close(compute_llama_inv_freq(config, 32), expected, rtol=0, atol=0)


def test_flexgen_rope_position_ids_cover_past_default_table():
    from bloombee.flexgen_utils.pytorch_backend import _DEFAULT_ROPE_TABLE_LEN, apply_rotary_emb, precompute_freqs_cis

    head_dim = 8
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = _DEFAULT_ROPE_TABLE_LEN + 17
    q = torch.randn(1, 1, 2, head_dim)
    k = torch.randn(1, 1, 2, head_dim)
    with_ids = precompute_freqs_cis(head_dim, _DEFAULT_ROPE_TABLE_LEN, inv_freq, position_ids=torch.tensor([[pos]]))
    q_out, k_out = apply_rotary_emb(q, k, with_ids)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    table = precompute_freqs_cis(head_dim, _DEFAULT_ROPE_TABLE_LEN, inv_freq, position_ids=None)
    empty = table[pos : pos + 1]
    assert empty.shape[0] == 0
    with pytest.raises(RuntimeError):
        apply_rotary_emb(q, k, empty)
