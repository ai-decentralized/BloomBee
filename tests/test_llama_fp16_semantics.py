"""Match HF rounding in the FlexGen operators used by Llama backward."""

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb

from bloombee.flexgen_utils.pytorch_backend import apply_rotary_emb, precompute_freqs_cis, rms_norm

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
def test_fp16_rmsnorm_output_and_input_gradient_match_hf(device):
    torch.manual_seed(937)
    layer = LlamaRMSNorm(64, eps=1e-5).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight) * 0.1 + 1.0)
    actual_input = torch.randn(2, 16, 64, dtype=torch.float16, device=device).requires_grad_(True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = rms_norm(actual_input, layer.weight, 1e-5)
    reference = layer(reference_input)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    projection = torch.randn_like(actual)
    (actual * projection).sum().backward()
    (reference * projection).sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fp16_rotary_output_and_input_gradients_match_hf(device, head_dim):
    torch.manual_seed(938)
    config = LlamaConfig(
        hidden_size=head_dim, num_attention_heads=1, num_key_value_heads=1, head_dim=head_dim, rope_theta=100000.0
    )
    rotary = LlamaRotaryEmbedding(config).to(device)
    query = torch.randn(2, 16, 9, head_dim, dtype=torch.float16, device=device).requires_grad_(True)
    key = torch.randn(2, 16, 3, head_dim, dtype=torch.float16, device=device).requires_grad_(True)
    reference_query = query.detach().clone().requires_grad_(True)
    reference_key = key.detach().clone().requires_grad_(True)
    positions = torch.arange(16, device=device).unsqueeze(0).expand(2, -1)
    cos, sin = rotary(reference_query, positions)
    expected_query, expected_key = apply_rotary_pos_emb(
        reference_query.transpose(1, 2), reference_key.transpose(1, 2), cos, sin
    )
    frequencies = precompute_freqs_cis(head_dim, 16, rotary.inv_freq)
    actual_query, actual_key = apply_rotary_emb(query, key, frequencies)
    torch.testing.assert_close(actual_query, expected_query.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key.transpose(1, 2), rtol=0, atol=0)
    (actual_query.float().square().mean() + actual_key.float().square().mean()).backward()
    (expected_query.float().square().mean() + expected_key.float().square().mean()).backward()
    torch.testing.assert_close(query.grad, reference_query.grad, rtol=0, atol=0)
    torch.testing.assert_close(key.grad, reference_key.grad, rtol=0, atol=0)
