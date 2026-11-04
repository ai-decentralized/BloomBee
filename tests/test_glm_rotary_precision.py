"""GLM-4.7-Flash RoPE frequencies must survive the server dtype cast.

Server loading ends in ``block.to(dtype)``. A bare decoder layer does not
inherit a full-model fp32 exclusion, so that cast used to store ``inv_freq``
in bf16/fp16. ``Glm4MoeLiteRotaryEmbedding`` promotes the buffer with
``.float()`` afterwards, which cannot restore the rounded mantissa. At a few
thousand tokens the resulting angle error is already tens of degrees.
"""

from copy import deepcopy

import pytest
import torch

from bloombee.models.glm4_moe_lite.block import WrappedGlm4MoeLiteBlock


def _mini_config():
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig

    return Glm4MoeLiteConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=1,
        num_experts_per_tok=1,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        max_position_embeddings=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        mlp_layer_types=["dense"],
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_glm_rotary_frequencies_and_long_positions_survive_cast(dtype):
    config = _mini_config()
    block = WrappedGlm4MoeLiteBlock(config, layer_idx=0).eval()
    reference = deepcopy(block._rotary_emb)
    originals = {name: getattr(block._rotary_emb, name).clone() for name in ("inv_freq", "original_inv_freq")}
    assert not torch.equal(originals["inv_freq"].to(dtype).float(), originals["inv_freq"])

    block.to(dtype)
    assert block.input_layernorm.weight.dtype == dtype
    for name, original in originals.items():
        value = getattr(block._rotary_emb, name)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, original, rtol=0, atol=0)

    # Evaluate the real rotary module at short and published long-context positions.
    positions = torch.tensor([[0, 2048, 8192, 202751]])
    inputs = torch.zeros(1, 4, config.hidden_size, dtype=dtype)
    with torch.inference_mode():
        actual = block._rotary_emb(inputs, positions)
        expected = reference(inputs, positions)
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device migration")
def test_glm_rotary_frequencies_survive_dtype_and_device_roundtrip():
    block = WrappedGlm4MoeLiteBlock(_mini_config(), layer_idx=0).eval()
    original = block._rotary_emb.inv_freq.clone()
    block.cuda().half()
    assert block._rotary_emb.inv_freq.device.type == "cuda"
    assert block._rotary_emb.original_inv_freq.device.type == "cuda"
    block.cpu().bfloat16()
    for name in ("inv_freq", "original_inv_freq"):
        assert getattr(block._rotary_emb, name).dtype == torch.float32
        torch.testing.assert_close(getattr(block._rotary_emb, name), original, rtol=0, atol=0)
