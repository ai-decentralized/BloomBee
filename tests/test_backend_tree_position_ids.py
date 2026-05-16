import torch

from bloombee.server.backend import TransformerBackend


def test_generation_tree_position_ids_follow_tree_depths():
    backend = object.__new__(TransformerBackend)

    # Cache already contains positions 0..35. The next root is position 36.
    kv_cache_position_ids = torch.tensor([[35]], dtype=torch.long)
    prefill_length = torch.tensor([36], dtype=torch.long)
    cache_len = 36
    target_seq_len = 4

    # Local rows after cache columns:
    # root, two depth-1 siblings, one depth-2 child.
    local_tree_mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    tree_attention_mask = torch.zeros((1, target_seq_len, cache_len + target_seq_len), dtype=torch.bool)
    tree_attention_mask[:, :, cache_len:] = local_tree_mask

    position_ids = backend._create_tree_position_ids_with_invalid_cache(
        width=1,
        depth=1,
        prefill_length=prefill_length,
        kv_cache_position_ids=kv_cache_position_ids,
        batch_offset=0,
        device=torch.device("cpu"),
        target_seq_len=target_seq_len,
        tree_attention_mask=tree_attention_mask,
        cache_len=cache_len,
    )

    assert position_ids.tolist() == [[36, 37, 37, 38]]


def test_generation_tree_position_ids_slice_full_batch_masks_for_microbatch():
    backend = object.__new__(TransformerBackend)

    kv_cache_position_ids = torch.tensor([[10], [20]], dtype=torch.long)
    prefill_length = torch.tensor([21], dtype=torch.long)
    cache_len = 21
    target_seq_len = 2
    tree_attention_mask = torch.zeros((2, target_seq_len, cache_len + target_seq_len), dtype=torch.bool)
    tree_attention_mask[:, 0, cache_len] = True
    tree_attention_mask[:, 1, cache_len:] = torch.tensor([True, True])

    position_ids = backend._create_tree_position_ids_with_invalid_cache(
        width=1,
        depth=1,
        prefill_length=prefill_length,
        kv_cache_position_ids=kv_cache_position_ids,
        batch_offset=1,
        device=torch.device("cpu"),
        target_seq_len=target_seq_len,
        tree_attention_mask=tree_attention_mask,
        cache_len=cache_len,
    )

    assert position_ids.tolist() == [[21, 22]]


def test_generation_tree_position_ids_use_logical_length_for_sparse_tree_cache():
    backend = object.__new__(TransformerBackend)

    # The accepted draft node may live in a sparse physical tree slot. The
    # attention mask must still read slot 77, but RoPE should continue at the
    # logical sequence position after root + one accepted draft token.
    kv_cache_position_ids = torch.tensor([[66, 77]], dtype=torch.long)
    prefill_length = torch.tensor([67], dtype=torch.long)
    cache_len = 78
    target_seq_len = 3

    local_tree_mask = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )
    tree_attention_mask = torch.zeros((1, target_seq_len, cache_len + target_seq_len), dtype=torch.bool)
    tree_attention_mask[:, :, cache_len:] = local_tree_mask

    position_ids = backend._create_tree_position_ids_with_invalid_cache(
        width=1,
        depth=1,
        prefill_length=prefill_length,
        kv_cache_position_ids=kv_cache_position_ids,
        batch_offset=0,
        device=torch.device("cpu"),
        target_seq_len=target_seq_len,
        tree_attention_mask=tree_attention_mask,
        cache_len=cache_len,
    )

    assert position_ids.tolist() == [[68, 69, 70]]


def test_chunked_prefill_kv_helpers_accumulate_chunks_in_sequence_order():
    batch_size = 2
    num_heads = 3
    head_dim = 2
    bh = batch_size * num_heads

    key1 = torch.arange(bh * head_dim * 4, dtype=torch.float32).view(bh, head_dim, 4)
    value1 = torch.arange(bh * 4 * head_dim, dtype=torch.float32).view(bh, 4, head_dim)
    key2 = torch.arange(bh * head_dim * 5, dtype=torch.float32).view(bh, head_dim, 5) + 1000
    value2 = torch.arange(bh * 5 * head_dim, dtype=torch.float32).view(bh, 5, head_dim) + 1000

    llama1 = TransformerBackend._kv_to_llama_layout((key1, value1), batch_size)
    llama2 = TransformerBackend._kv_to_llama_layout((key2, value2), batch_size)
    combined_llama = TransformerBackend._concat_llama_kvs(llama1, llama2)
    combined_bloom = TransformerBackend._concat_bloom_kv_chunks([(key1, value1), (key2, value2)])

    assert combined_llama[0].shape == (batch_size, num_heads, 9, head_dim)
    assert combined_llama[1].shape == (batch_size, num_heads, 9, head_dim)
    assert torch.equal(combined_llama[0][:, :, :4], llama1[0])
    assert torch.equal(combined_llama[0][:, :, 4:], llama2[0])
    assert torch.equal(combined_bloom[0], torch.cat([key1, key2], dim=2))
    assert torch.equal(combined_bloom[1], torch.cat([value1, value2], dim=1))

    roundtrip_bloom = TransformerBackend._kv_to_bloom_layout(combined_llama, batch_size)
    assert torch.equal(roundtrip_bloom[0], combined_bloom[0])
    assert torch.equal(roundtrip_bloom[1], combined_bloom[1])
