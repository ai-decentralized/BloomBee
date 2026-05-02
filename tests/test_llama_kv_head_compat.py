from types import SimpleNamespace

import torch

from bloombee.models.llama.block import WrappedLlamaBlock


def test_llama_cache_reorder_infers_kv_heads_from_groups():
    batch_size = 2
    num_heads = 8
    num_key_value_groups = 2
    num_key_value_heads = num_heads // num_key_value_groups
    seq_length = 3
    head_dim = 5

    block = WrappedLlamaBlock.__new__(WrappedLlamaBlock)
    block.self_attn = SimpleNamespace(
        num_heads=num_heads,
        num_key_value_groups=num_key_value_groups,
        head_dim=head_dim,
    )

    key_bloom = torch.arange(
        batch_size * num_key_value_heads * head_dim * seq_length,
        dtype=torch.float32,
    ).view(batch_size * num_key_value_heads, head_dim, seq_length)
    value_bloom = torch.arange(
        batch_size * num_key_value_heads * seq_length * head_dim,
        dtype=torch.float32,
    ).view(batch_size * num_key_value_heads, seq_length, head_dim)

    key_llama, value_llama = block._reorder_cache_from_bloom_to_llama(
        (key_bloom, value_bloom),
        batch_size=batch_size,
        seq_length=seq_length,
    )

    assert key_llama.shape == (batch_size, num_key_value_heads, seq_length, head_dim)
    assert value_llama.shape == (batch_size, num_key_value_heads, seq_length, head_dim)
    torch.testing.assert_close(
        key_llama.view(batch_size * num_key_value_heads, seq_length, head_dim),
        key_bloom.permute(0, 2, 1),
    )
    torch.testing.assert_close(
        value_llama.view(batch_size * num_key_value_heads, seq_length, head_dim),
        value_bloom,
    )

    key_roundtrip, value_roundtrip = block._reorder_cache_from_llama_to_bloom(
        (key_llama, value_llama),
        batch_size=batch_size,
        seq_length=seq_length,
    )
    torch.testing.assert_close(key_roundtrip, key_bloom)
    torch.testing.assert_close(value_roundtrip, value_bloom)
