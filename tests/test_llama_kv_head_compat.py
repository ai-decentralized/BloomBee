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


def test_llama_4d_gqa_broadcast_past_keeps_attention_heads():
    """FlexGen broadcasts GQA KV to H; chunked prefill past is [B, H, S, D].

    Slicing that past to n_kv makes cache_read_buf (S, B*n_kv, D) while
    mha_gen_llama writes (tgt, B*H, D), which crashes torch.cat on the next
    chunk of a long Llama-3 prompt.
    """
    batch_size = 1
    num_heads = 32
    num_key_value_heads = 8
    seq_length = 4
    head_dim = 4

    block = WrappedLlamaBlock.__new__(WrappedLlamaBlock)
    block.self_attn = SimpleNamespace(
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        num_key_value_groups=num_heads // num_key_value_heads,
        head_dim=head_dim,
    )

    key_4d = torch.arange(batch_size * num_heads * seq_length * head_dim, dtype=torch.float32).view(
        batch_size, num_heads, seq_length, head_dim
    )
    value_4d = key_4d + 0.5

    key_out, value_out = block._reorder_cache_from_bloom_to_llama(
        (key_4d, value_4d),
        batch_size=batch_size,
        seq_length=seq_length,
    )

    assert key_out.shape == (batch_size, num_heads, seq_length, head_dim)
    assert value_out.shape == (batch_size, num_heads, seq_length, head_dim)
    torch.testing.assert_close(key_out, key_4d)
    torch.testing.assert_close(value_out, value_4d)

    past_k = key_out.permute(2, 0, 1, 3).reshape(seq_length, batch_size * num_heads, head_dim)
    past_v = value_out.permute(2, 0, 1, 3).reshape(seq_length, batch_size * num_heads, head_dim)
    assert past_k.shape == (seq_length, batch_size * num_heads, head_dim)
    assert past_v.shape == (seq_length, batch_size * num_heads, head_dim)
    new_k = torch.zeros(1, batch_size * num_heads, head_dim)
    torch.cat([past_k, new_k], dim=0)
