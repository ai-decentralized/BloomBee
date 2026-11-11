"""Nemotron-H KV reads must keep packed key/value heads when the batch is larger than 1."""

from types import SimpleNamespace

import pytest
import torch

from bloombee.flexgen_utils.pytorch_backend import TorchTensor


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_nemotron_select_cache_keeps_packed_kv_heads(batch, make_cpu_cache_manager):
    """Batched Nemotron decode must see [B, num_key_value_heads, S, D].

    Query heads are 32 and KV heads are 2, so batch 2 is 4 packed rows.
    Viewing those rows with num_attention_heads yields [1, 4, S, D] (or
    [1, 32, S, D] once the row count reaches 32) and the next token's cache
    update crashes.
    """
    query_heads, kv_heads, seq, dim = 32, 2, 3, 4
    config = SimpleNamespace(
        model_type="nemotron_h",
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        hidden_size=query_heads * dim,
    )
    manager = make_cpu_cache_manager(config)
    rows = batch * kv_heads
    values = torch.arange(seq * rows * dim, dtype=torch.float32).view(seq, rows, dim)
    caches = (
        (
            TorchTensor.create_from_torch(values, manager.env.cpu),
            TorchTensor.create_from_torch(values + 10_000, manager.env.cpu),
        ),
    )
    k_pkv, v_pkv, _ = manager.select_cache(
        prefix_length=seq,
        cache_tensors=caches,
        requested_rows=batch,
    )
    assert k_pkv.shape == (batch, kv_heads, seq, dim)
    assert v_pkv.shape == (batch, kv_heads, seq, dim)
    expected = values.view(seq, batch, kv_heads, dim).permute(1, 2, 0, 3)
    torch.testing.assert_close(k_pkv, expected)
    torch.testing.assert_close(v_pkv, expected + 10_000)


def test_attention_head_slab_select_cache_unchanged(make_cpu_cache_manager):
    """Non-Nemotron slabs stay strided by query heads."""
    batch, heads, seq, dim = 2, 8, 3, 4
    config = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=2, num_key_value_groups=4, hidden_size=32)
    manager = make_cpu_cache_manager(config)
    rows = batch * heads
    values = torch.arange(seq * rows * dim, dtype=torch.float32).view(seq, rows, dim)
    caches = (
        (
            TorchTensor.create_from_torch(values, manager.env.cpu),
            TorchTensor.create_from_torch(values + 5, manager.env.cpu),
        ),
    )
    k_pkv, v_pkv, _ = manager.select_cache(
        prefix_length=seq,
        cache_tensors=caches,
        requested_rows=batch,
    )
    assert k_pkv.shape == (batch, heads, seq, dim)
    expected = values.view(seq, batch, heads, dim).permute(1, 2, 0, 3)
    torch.testing.assert_close(k_pkv, expected)
    torch.testing.assert_close(v_pkv, expected + 5)


@pytest.mark.parametrize("offset,rows,full_batch", [(1, 1, 4), (2, 2, 4), (0, 2, 0)])
def test_nemotron_microbatch_and_compacted_reads(offset, rows, full_batch, make_cpu_cache_manager):
    config = SimpleNamespace(model_type="nemotron_h", num_attention_heads=32, num_key_value_heads=2, hidden_size=64)
    manager = make_cpu_cache_manager(config)
    values = torch.arange(3 * 4 * 2 * 2, dtype=torch.float32).view(3, 8, 2)
    caches = (
        (
            TorchTensor.create_from_torch(values, manager.env.cpu),
            TorchTensor.create_from_torch(values + 1000, manager.env.cpu),
        ),
    )
    key, value, _ = manager.select_cache(
        3,
        batch_offset=offset,
        full_batch_size=full_batch,
        micro_batch_size=rows if full_batch else 0,
        requested_rows=rows,
        cache_tensors=caches,
    )
    expected = values[:, offset * 2 : (offset + rows) * 2].reshape(3, rows, 2, 2).permute(1, 2, 0, 3)
    torch.testing.assert_close(key, expected)
    torch.testing.assert_close(value, expected + 1000)


@pytest.mark.parametrize("batch", [2, 16])
def test_nemotron_batched_decode_matches_full_prefill(batch, make_cpu_cache_manager):
    from transformers.models.nemotron_h import NemotronHConfig

    from bloombee.models.nemotron_h.block import WrappedNemotronHBlock

    torch.manual_seed(7)
    config = NemotronHConfig(
        hidden_size=64,
        intermediate_size=128,
        layers_block_type=["full_attention"],
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=2,
        use_mamba_kernels=False,
    )
    config._attn_implementation = "eager"
    block = WrappedNemotronHBlock(config, layer_idx=0).eval()
    manager = make_cpu_cache_manager(config)
    slab = torch.zeros(4, batch * 2, 2)
    caches = (
        (
            TorchTensor.create_from_torch(slab.clone(), manager.env.cpu),
            TorchTensor.create_from_torch(slab.clone(), manager.env.cpu),
        ),
    )
    inputs = torch.randn(batch, 4, 64)
    mask = torch.triu(torch.full((batch, 4, 4), float("-inf")), diagonal=1)
    with torch.inference_mode():
        expected, full_kv = block(inputs, attention_mask=mask, use_cache=True)
        _, prefix = block(inputs[:, :3], attention_mask=mask[:, :3, :3], use_cache=True)
        manager._write_kvs(
            prefix, start_position=0, full_batch_size=batch, micro_batch_size=batch, cache_tensors=caches
        )
        key, value, _ = manager.select_cache(3, requested_rows=batch, cache_tensors=caches)
        decoded, new_kv = block(inputs[:, 3:], layer_past=(key, value), attention_mask=mask[:, 3:], use_cache=True)
        manager._write_kvs(
            new_kv, start_position=3, full_batch_size=batch, micro_batch_size=batch, cache_tensors=caches
        )
        key, value, _ = manager.select_cache(4, requested_rows=batch, cache_tensors=caches)
    torch.testing.assert_close(decoded, expected[:, 3:], atol=1e-5, rtol=1e-5)
    expected_key = full_kv[0].transpose(1, 2).reshape(batch, 2, 4, 2)
    expected_value = full_kv[1].reshape(batch, 2, 4, 2)
    torch.testing.assert_close(key, expected_key, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(value, expected_value, atol=1e-5, rtol=1e-5)
