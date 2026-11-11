"""Check Nemotron-H's BloomBee mask against an explicit attention-head axis."""

import pytest
import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from bloombee.models.nemotron_h.block import WrappedNemotronHBlock


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("past_length", [0, 5])
@torch.inference_mode()
def test_batched_mask_matches_explicit_head_axis(implementation, batch, past_length):
    torch.manual_seed(0)
    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        layers_block_type=["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=4,
        n_groups=1,
        conv_kernel=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = implementation
    block = WrappedNemotronHBlock(config, layer_idx=0).eval()
    query_length = 1 if past_length else 4
    hidden = torch.randn(batch, query_length, config.hidden_size)
    allowed = torch.ones(batch, query_length, past_length + query_length, dtype=torch.bool)
    if not past_length:
        allowed = allowed.tril()
    # Different padding per row exposes silent batch/head broadcasting at B=H.
    for row in range(1, batch):
        allowed[row, :, 1 : 1 + row] = False
    mask = torch.full(allowed.shape, -65504.0).masked_fill(allowed, 0.0)
    layer_past = None
    if past_length:
        layer_past = (
            torch.randn(batch * config.num_key_value_heads, config.head_dim, past_length),
            torch.randn(batch * config.num_key_value_heads, past_length, config.head_dim),
        )

    expected, expected_kv = block(hidden, attention_mask=mask.unsqueeze(1), layer_past=layer_past, use_cache=True)
    actual, actual_kv = block(hidden, attention_mask=mask, layer_past=layer_past, use_cache=True)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_kv, expected_kv)
    key, value = actual_kv
    assert key.shape == (batch * config.num_key_value_heads, config.head_dim, query_length)
    assert value.shape == (batch * config.num_key_value_heads, query_length, config.head_dim)
