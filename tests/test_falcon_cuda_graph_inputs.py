import pytest
import torch
from transformers import FalconConfig

from bloombee.models.falcon.block import OptimizedFalconDecoderLayer, OptimizedFalconRotaryEmbedding, apply_rotary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph regression")


def config():
    return FalconConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        num_hidden_layers=1,
        new_decoder_architecture=True,
        multi_query=False,
        hidden_dropout=0,
        attention_dropout=0,
    )


def test_rotary_graph_owns_inputs_and_handles_batch_changes():
    module = OptimizedFalconRotaryEmbedding(16).cuda().half()
    with torch.inference_mode():
        for batch, position in [(1, 32), (1, 33), (2, 64), (1, 12)]:
            query = torch.randn(batch * 4, 1, 16, device="cuda", dtype=torch.float16)
            key = torch.randn(batch * 2, 1, 16, device="cuda", dtype=torch.float16)
            cos, sin = module.cos_sin(1, position, query.device, query.dtype)
            before_cos, before_sin = module.cos_cached.clone(), module.sin_cached.clone()
            actual = module(query, key, position)
            expected = apply_rotary(query, key, cos, sin)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(module.cos_cached, before_cos, rtol=0, atol=0)
            torch.testing.assert_close(module.sin_cached, before_sin, rtol=0, atol=0)


def test_layernorm_and_split_graphs_handle_batch_changes():
    layer = OptimizedFalconDecoderLayer(config()).cuda().half().eval()
    attention = layer.self_attention
    with torch.inference_mode():
        for batch in [1, 2, 1]:
            hidden = torch.randn(batch, 1, 64, device="cuda", dtype=torch.float16)
            original = hidden.clone()
            outputs = layer._optimized_apply_ln(hidden)
            for actual, expected in zip(outputs, (layer.ln_attn(hidden), layer.ln_mlp(hidden))):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(hidden, original, rtol=0, atol=0)
            qkv = attention.query_key_value(hidden)
            original_qkv = qkv.clone()
            for actual, expected in zip(attention._optimized_split_heads(qkv), attention._split_heads(qkv)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(qkv, original_qkv, rtol=0, atol=0)
