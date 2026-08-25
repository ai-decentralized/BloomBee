from types import SimpleNamespace

import torch

from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor
from bloombee.server.memory_cache_manager import KVCacheManager


def _make_manager() -> KVCacheManager:
    cpu_device = TorchDevice("cpu")
    env = ExecutionEnv(gpu=cpu_device, cpu=cpu_device, disk=None, mixed=None)
    policy = Policy(
        1,
        1,
        100,
        0,
        0,
        100,
        100,
        0,
        overlap=False,
        sep_layer=True,
        pin_weight=False,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        compress_cache=False,
        comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )
    block_config = SimpleNamespace(
        num_attention_heads=4,
        hidden_size=32,
        num_key_value_groups=1,
    )
    return KVCacheManager(1024, None, policy, env, block_config)


def test_select_cache_without_reorder_compacts_sparse_tree_positions():
    manager = _make_manager()

    batch_size = 2
    num_heads = manager.block_config.num_attention_heads
    head_dim = 3
    seq_len = 10
    bh = batch_size * num_heads

    k_cache = torch.arange(seq_len * bh * head_dim, dtype=torch.float32).view(seq_len, bh, head_dim)
    v_cache = k_cache + 10000
    cache_tensors = (
        (
            TorchTensor.create_from_torch(k_cache, manager.attention_compute),
            TorchTensor.create_from_torch(v_cache, manager.attention_compute),
        ),
    )

    kv_cache_position_ids = torch.tensor(
        [
            [2, 5, 7],
            [2, 3, -1],
        ],
        dtype=torch.long,
    )

    k_pkv, v_pkv, cache_len = manager.select_cache_without_reorder(
        kv_cache_position_ids,
        cache_tensors=cache_tensors,
    )

    assert cache_len == 5
    assert k_pkv.shape == (batch_size, num_heads, 5, head_dim)
    assert v_pkv.shape == (batch_size, num_heads, 5, head_dim)

    raw_k = k_cache.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    raw_v = v_cache.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)

    assert torch.equal(k_pkv[0, :, :2], raw_k[0, :, :2])
    assert torch.equal(k_pkv[0, :, 2], raw_k[0, :, 2])
    assert torch.equal(k_pkv[0, :, 3], raw_k[0, :, 5])
    assert torch.equal(k_pkv[0, :, 4], raw_k[0, :, 7])
    assert torch.equal(v_pkv[0, :, 3], raw_v[0, :, 5])

    assert torch.equal(k_pkv[1, :, :4], raw_k[1, :, :4])
    assert torch.equal(v_pkv[1, :, :4], raw_v[1, :, :4])
    assert torch.equal(k_pkv[1, :, 4], torch.zeros_like(k_pkv[1, :, 4]))


def test_fast_spec_cache_update_compacts_sparse_path_and_writes_tree_per_row():
    manager = _make_manager()

    batch_size = 2
    num_heads = manager.block_config.num_attention_heads
    head_dim = 3
    seq_len = 32
    tree_len = 3
    bh = batch_size * num_heads

    k_cache = torch.arange(seq_len * bh * head_dim, dtype=torch.float32).view(seq_len, bh, head_dim)
    v_cache = k_cache + 10000
    original_k = k_cache.clone()
    original_v = v_cache.clone()
    cache_tensors = (
        (
            TorchTensor.create_from_torch(k_cache, manager.attention_compute),
            TorchTensor.create_from_torch(v_cache, manager.attention_compute),
        ),
    )

    kv_cache_position_ids = torch.tensor(
        [
            [4, 7, 9],
            [4, 5, -1],
        ],
        dtype=torch.long,
    )
    compact_lengths = torch.tensor([7, 6], dtype=torch.long)

    key = torch.arange(bh * head_dim * tree_len, dtype=torch.float32).view(bh, head_dim, tree_len) + 50000
    value = torch.arange(bh * tree_len * head_dim, dtype=torch.float32).view(bh, tree_len, head_dim) + 60000

    assert manager._try_fast_spec_cache_update(
        new_kvs=(key, value),
        kv_cache_position_ids=kv_cache_position_ids,
        compact_lengths=compact_lengths,
        cache_tensors=cache_tensors,
        batch_offset=0,
        full_batch_size=0,
        micro_batch_size=0,
        cache_manager=manager,
    )

    raw_new_k = key.permute(2, 0, 1)
    raw_new_v = value.permute(1, 0, 2)

    # Prefix before the previous root is untouched.
    torch.testing.assert_close(k_cache[:4], original_k[:4])
    torch.testing.assert_close(v_cache[:4], original_v[:4])

    # Row 0 accepted sparse slots [4, 7, 9] are compacted to [4, 5, 6].
    torch.testing.assert_close(k_cache[4:7, 0:num_heads], original_k[[4, 7, 9], 0:num_heads])
    torch.testing.assert_close(v_cache[4:7, 0:num_heads], original_v[[4, 7, 9], 0:num_heads])

    # Row 1 accepted slots [4, 5] are already compact and remain in place.
    row1_bh = slice(num_heads, 2 * num_heads)
    torch.testing.assert_close(k_cache[4:6, row1_bh], original_k[4:6, row1_bh])
    torch.testing.assert_close(v_cache[4:6, row1_bh], original_v[4:6, row1_bh])

    # The new root+tree is written after each row's own compact length.
    torch.testing.assert_close(k_cache[7:10, 0:num_heads], raw_new_k[:, 0:num_heads])
    torch.testing.assert_close(v_cache[7:10, 0:num_heads], raw_new_v[:, 0:num_heads])
    torch.testing.assert_close(k_cache[6:9, row1_bh], raw_new_k[:, row1_bh])
    torch.testing.assert_close(v_cache[6:9, row1_bh], raw_new_v[:, row1_bh])

    # Next-step selection can compact accepted tokens from the freshly written
    # tree without depending on a full prefix reorder.
    next_positions = torch.tensor(
        [
            [7, 9, -1],
            [6, 7, 8],
        ],
        dtype=torch.long,
    )
    k_pkv, v_pkv, cache_len = manager.select_cache_without_reorder(
        next_positions,
        cache_tensors=cache_tensors,
    )
    assert cache_len == 9
    raw_k = k_cache.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    raw_v = v_cache.view(seq_len, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    torch.testing.assert_close(k_pkv[0, :, 8], raw_k[0, :, 9])
    torch.testing.assert_close(v_pkv[0, :, 8], raw_v[0, :, 9])
    torch.testing.assert_close(k_pkv[1, :, 6:9], raw_k[1, :, 6:9])
