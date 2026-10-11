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
