"""Tests for DeepSeek-V3 int8 expert weight quantization.

DeepseekV3Experts batches all routed experts into two 3D nn.Parameter tensors
(gate_up_proj, down_proj) rather than per-expert nn.Linear submodules, so this
quantizes those tensors directly instead of using bitsandbytes' nn.Linear
replacements. See bloombee/models/deepseekv3/{expert_quant,quantized_experts}.py
and the quantization branch in convert_block().
"""
import pytest
import torch

from bloombee.models.deepseekv3.block import WrappedDeepseekV3Block
from bloombee.models.deepseekv3.config import DistributedDeepseekV3Config
from bloombee.models.deepseekv3.expert_quant import dequantize_expert_weight, quantize_expert_weight
from bloombee.models.deepseekv3.quantized_experts import QuantizedDeepseekV3Experts
from bloombee.models.mixtral.block import WrappedMixtralBlock
from bloombee.utils.convert_block import QuantType, convert_block


def _make_config(n_routed_experts=4):
    cfg = DistributedDeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=n_routed_experts,
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
    block = WrappedDeepseekV3Block(cfg, layer_idx=layer_idx)
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return block.eval()


# --- expert_quant.py: quantize/dequantize round trip -----------------------------


@pytest.mark.parametrize("shape", [(6, 40, 256), (3, 8, 32)])  # second shape doesn't divide 128 evenly
def test_quantize_dequantize_round_trip_within_int8_noise_floor(shape):
    torch.manual_seed(0)
    num_experts, out_f, in_f = shape
    weight = torch.randn(num_experts, out_f, in_f) * 0.02

    data, scale, group_size = quantize_expert_weight(weight, group_size=128)
    assert data.dtype == torch.int8
    assert data.shape == weight.shape

    for e in range(num_experts):
        deq = dequantize_expert_weight(data, scale, e, group_size, torch.float32)
        rel_err = (deq - weight[e]).abs().max() / weight[e].abs().max()
        assert rel_err < 0.01  # int8 symmetric quant: worst case ~1/127 ~= 0.8%


def test_dequantize_accepts_tensor_expert_idx():
    torch.manual_seed(0)
    weight = torch.randn(4, 8, 256) * 0.02
    data, scale, group_size = quantize_expert_weight(weight, group_size=128)

    by_int = dequantize_expert_weight(data, scale, 2, group_size, torch.float32)
    by_tensor = dequantize_expert_weight(data, scale, torch.tensor(2), group_size, torch.float32)
    torch.testing.assert_close(by_int, by_tensor)


# --- QuantizedDeepseekV3Experts: parity against HF's DeepseekV3MoE ---------------


def test_quantized_experts_matches_reference_moe_within_tolerance():
    from transformers.models.deepseek_v3 import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    torch.manual_seed(0)
    cfg = DeepseekV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
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
    )
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"

    moe = DeepseekV3MoE(cfg).eval()
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.02)

    x = torch.randn(1, 5, cfg.hidden_size) * 0.02
    ref_out = moe(x)

    moe.experts = QuantizedDeepseekV3Experts(moe.experts, group_size=128)
    q_out = moe(x)

    assert torch.isfinite(q_out).all()
    rel_err = (ref_out - q_out).abs().max() / ref_out.abs().max()
    assert rel_err < 0.02


# --- convert_block(): wiring, including the non-MoE and non-DeepSeek-V3 paths ----


def test_convert_block_int8_swaps_moe_experts_and_preserves_output():
    torch.manual_seed(0)
    cfg = _make_config()
    block = _new_block(cfg, layer_idx=1)  # MoE layer

    h = torch.randn(1, 5, cfg.hidden_size)
    ref_out, _ = block(h, use_cache=False)

    device = torch.device("cpu")
    convert_block(
        block, 1, cfg, tensor_parallel_devices=(device,), output_device=device, quant_type=QuantType.INT8, policy=None
    )
    assert isinstance(block.mlp.experts, QuantizedDeepseekV3Experts)

    q_out, _ = block(h, use_cache=False)
    assert torch.isfinite(q_out).all()
    rel_err = (ref_out - q_out).abs().max() / ref_out.abs().max()
    assert rel_err < 0.02


def test_convert_block_int8_leaves_dense_layer_untouched():
    torch.manual_seed(0)
    cfg = _make_config()
    block = _new_block(cfg, layer_idx=0)  # dense layer (< first_k_dense_replace), no .experts

    device = torch.device("cpu")
    convert_block(
        block, 0, cfg, tensor_parallel_devices=(device,), output_device=device, quant_type=QuantType.INT8, policy=None
    )
    assert not hasattr(block.mlp, "experts")


def test_convert_block_int8_rejects_non_deepseek_models():
    class FakeConfig:
        block_class = WrappedMixtralBlock
        model_type = "mixtral"

    device = torch.device("cpu")
    with pytest.raises(NotImplementedError):
        convert_block(
            torch.nn.Linear(2, 2),
            0,
            FakeConfig(),
            tensor_parallel_devices=(device,),
            output_device=device,
            quant_type=QuantType.INT8,
            policy=None,
        )


def test_convert_block_none_does_not_touch_experts():
    torch.manual_seed(0)
    cfg = _make_config()
    block = _new_block(cfg, layer_idx=1)
    original_experts = block.mlp.experts

    device = torch.device("cpu")
    convert_block(
        block, 1, cfg, tensor_parallel_devices=(device,), output_device=device, quant_type=QuantType.NONE, policy=None
    )
    assert block.mlp.experts is original_experts
