"""Exercise serving cache layouts and FlexGen kernels with real tensors."""

from types import SimpleNamespace

import pytest
import torch

from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor
from bloombee.models.falcon.block import WrappedFalconBlock
from bloombee.models.llama.block import WrappedLlamaBlock
from bloombee.server.backend import TransformerBackend
from bloombee.server.memory_cache_manager import KVCacheManager


def _cpu_cache_manager():
    device = TorchDevice("cpu")
    env = ExecutionEnv(gpu=device, cpu=device)
    compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(
        gpu_batch_size=1,
        num_gpu_batches=1,
        w_gpu_percent=100,
        w_cpu_percent=0,
        cache_gpu_percent=0,
        cache_cpu_percent=100,
        act_gpu_percent=100,
        act_cpu_percent=0,
        overlap=False,
        sep_layer=True,
        pin_weight=False,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=compression,
        compress_cache=False,
        comp_cache_config=compression,
    )
    config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, num_key_value_groups=4, hidden_size=32)
    return KVCacheManager(1024, None, policy, env, config)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_batch_hint", [False, True])
def test_gqa_fullbatch_write_preserves_values_and_batch_stride(batch, packed, with_batch_hint):
    manager = _cpu_cache_manager()
    heads, kv_heads, dim, tokens = 8, 2, 4, 2
    source_heads = kv_heads if packed else heads
    key = torch.arange(batch * source_heads * dim * tokens, dtype=torch.float32).view(batch * source_heads, dim, tokens)
    value = key.transpose(1, 2).contiguous() + 1000
    key_slab = torch.full((4, batch * heads, dim), -1.0)
    value_slab = torch.full_like(key_slab, -1.0)
    caches = (
        (
            TorchTensor.create_from_torch(key_slab, manager.env.cpu),
            TorchTensor.create_from_torch(value_slab, manager.env.cpu),
        ),
    )
    try:
        manager._write_kvs(
            (key, value),
            start_position=1,
            full_batch_size=batch if with_batch_hint else 0,
            micro_batch_size=batch if with_batch_hint else 0,
            cache_tensors=caches,
        )
        expected_key, expected_value = torch.full_like(key_slab, -1.0), torch.full_like(value_slab, -1.0)
        for row in range(batch):
            src = slice(row * source_heads, (row + 1) * source_heads)
            dst = slice(row * heads, row * heads + source_heads)
            expected_key[1:3, dst] = key[src].permute(2, 0, 1)
            expected_value[1:3, dst] = value[src].permute(1, 0, 2)
        torch.testing.assert_close(key_slab, expected_key)
        torch.testing.assert_close(value_slab, expected_value)
    finally:
        manager._reorder_executor.shutdown(wait=True)


def test_gqa_write_does_not_override_an_explicit_packed_batch_hint():
    manager = _cpu_cache_manager()
    # Four packed sequences have eight source rows, but only one sequence
    # fits in this slab. Their metadata must not be reinterpreted as B=1 MHA.
    key = torch.ones(8, 4, 1)
    value = key.transpose(1, 2).contiguous()
    slab = torch.zeros(4, 8, 4)
    caches = (
        (
            TorchTensor.create_from_torch(slab.clone(), manager.env.cpu),
            TorchTensor.create_from_torch(slab.clone(), manager.env.cpu),
        ),
    )
    try:
        with pytest.raises((AssertionError, RuntimeError)):
            manager._write_kvs(
                (key, value), start_position=0, full_batch_size=4, micro_batch_size=4, cache_tensors=caches
            )
    finally:
        manager._reorder_executor.shutdown(wait=True)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize(
    "architecture,heads,kv_heads,multi_query", [(True, 8, 2, False), (False, 7, 1, True), (False, 8, 8, False)]
)
def test_falcon_slab_reorder_ignores_padding_and_preserves_batch(batch, architecture, heads, kv_heads, multi_query):
    config = SimpleNamespace(
        new_decoder_architecture=architecture,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        num_key_value_groups=heads // kv_heads,
        multi_query=multi_query,
    )
    block = WrappedFalconBlock.__new__(WrappedFalconBlock)
    torch.nn.Module.__init__(block)
    block.config = config
    packed = torch.arange(batch * kv_heads * 3 * 4, dtype=torch.float32).view(batch, kv_heads, 3, 4)
    key_slab = torch.full((batch, heads, 3, 4), -1000.0)
    value_slab = torch.full_like(key_slab, -1000.0)
    key_slab[:, :kv_heads] = packed
    value_slab[:, :kv_heads] = packed + 1000
    key, value = block._reorder_cache_from_bloom_to_falcon((key_slab, value_slab))
    expected = packed.repeat_interleave(heads // kv_heads, dim=1) if architecture else packed
    torch.testing.assert_close(key, expected.reshape(-1, 3, 4))
    torch.testing.assert_close(value, (expected + 1000).reshape(-1, 3, 4))


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (7, 1), (8, 8)])
def test_chunked_prefill_concat_preserves_prefix_and_new_packed_kv(batch, heads, kv_heads):
    prefix = torch.arange(batch * heads * 3 * 4, dtype=torch.float32).view(batch, heads, 3, 4)
    new_key = torch.arange(batch * kv_heads * 4 * 2, dtype=torch.float32).view(batch * kv_heads, 4, 2)
    new_value = new_key.transpose(1, 2).contiguous() + 1000
    right = TransformerBackend._kv_to_llama_layout((new_key, new_value), batch)
    key, value = TransformerBackend._concat_llama_kvs((prefix, prefix + 1000), right)
    expected_key = torch.cat([prefix[:, :kv_heads], right[0]], dim=2)
    expected_value = torch.cat([prefix[:, :kv_heads] + 1000, right[1]], dim=2)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


@pytest.mark.parametrize("left_heads,right_heads,value_heads", [(2, 8, 8), (8, 3, 3), (8, 2, 1), (0, 2, 2)])
def test_chunked_prefill_concat_rejects_invalid_head_layouts(left_heads, right_heads, value_heads):
    left = (torch.zeros(1, left_heads, 3, 4), torch.zeros(1, left_heads, 3, 4))
    right = (torch.zeros(1, right_heads, 2, 4), torch.zeros(1, value_heads, 2, 4))
    with pytest.raises(RuntimeError, match="incompatible head layouts"):
        TransformerBackend._concat_llama_kvs(left, right)


def test_chunked_prefill_concat_rejects_mixed_ranks():
    left = (torch.zeros(1, 8, 3, 4), torch.zeros(1, 8, 3, 4))
    right = (torch.zeros(8, 4, 2), torch.zeros(8, 2, 4))
    with pytest.raises(RuntimeError, match="mixed ranks"):
        TransformerBackend._concat_llama_kvs(left, right)


def _flexgen_weights(device):
    def wrap(tensor):
        return TorchTensor.create_from_torch(tensor.to(device.dev), device)

    return dict(
        w_q=wrap(torch.randn(16, 16, dtype=torch.float16) * 0.1),
        w_k=wrap(torch.randn(8, 16, dtype=torch.float16) * 0.1),
        w_v=wrap(torch.randn(8, 16, dtype=torch.float16) * 0.1),
        w_out=wrap(torch.randn(16, 16, dtype=torch.float16) * 0.1),
        input_layernorm=wrap(torch.ones(16, dtype=torch.float16)),
        rotary_emb_inv_freq=wrap(torch.tensor([1.0, 0.01], dtype=torch.float32)),
        donate=(False, False),
        compress_cache=False,
        comp_config=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexGen decode kernel uses CUDA output")
@pytest.mark.parametrize("prefix", [3, 4096, 5000])
def test_flexgen_decode_sequential_rope_matches_explicit_positions(prefix):
    torch.manual_seed(0)
    device = TorchDevice("cuda:0")
    weights = _flexgen_weights(device)
    hidden = torch.randn(2, 2, 16, device=device.dev, dtype=torch.float16)
    past_key = torch.randn(prefix, 8, 4, device=device.dev, dtype=torch.float16)
    past_value = torch.randn_like(past_key)
    mask = torch.ones(2, 1, 2, prefix + 2, device=device.dev, dtype=torch.bool)
    mask[:, :, 0, -1] = False
    positions = torch.arange(prefix, prefix + 2, device=device.dev).expand(2, -1)
    results = []
    for position_ids in (None, positions):
        result = device.mha_gen_llama(
            TorchTensor.create_from_torch(hidden, device),
            TorchTensor.create_from_torch(mask, device),
            **weights,
            n_head=4,
            k_cache=TorchTensor.create_from_torch(past_key.clone(), device),
            v_cache=TorchTensor.create_from_torch(past_value.clone(), device),
            attn_sparsity=1.0,
            rotary_position_ids=position_ids,
        )
        results.append(tuple(tensor.data.clone() for tensor in result))
    for sequential, explicit in zip(*results):
        torch.testing.assert_close(sequential, explicit, atol=0.003, rtol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexGen decode kernel uses CUDA output")
def test_flexgen_gqa_chunked_prefill_matches_unsplit_attention():
    torch.manual_seed(1)
    device = TorchDevice("cuda:0")
    weights = _flexgen_weights(device)
    batch, length, split = 2, 9, 5
    hidden = torch.randn(batch, length, 16, device=device.dev, dtype=torch.float16)
    mask = torch.ones(batch, 1, length, length, device=device.dev, dtype=torch.bool).tril()
    scores = torch.zeros_like(mask, dtype=torch.float16).masked_fill(~mask, float("-inf"))
    positions = torch.arange(length, device=device.dev).expand(batch, -1)
    full = device.mha_llama(
        TorchTensor.create_from_torch(hidden, device),
        TorchTensor.create_from_torch(scores, device),
        **weights,
        num_attention_heads=4,
        rotary_position_ids=positions,
    )
    first = device.mha_llama(
        TorchTensor.create_from_torch(hidden[:, :split], device),
        TorchTensor.create_from_torch(scores[:, :, :split, :split], device),
        **weights,
        num_attention_heads=4,
        rotary_position_ids=positions[:, :split],
    )
    # Follow the backend's conversion into 4D past, then the Llama wrapper.
    bloom = (first[1].data.permute(1, 2, 0), first[2].data.permute(1, 0, 2))
    past = TransformerBackend._kv_to_llama_layout(bloom, batch)
    block = WrappedLlamaBlock.__new__(WrappedLlamaBlock)
    block.self_attn = SimpleNamespace(num_heads=4, num_key_value_heads=2, num_key_value_groups=2, head_dim=4)
    key, value = block._reorder_cache_from_bloom_to_llama(past, batch, split)
    second = device.mha_gen_llama(
        TorchTensor.create_from_torch(hidden[:, split:], device),
        TorchTensor.create_from_torch(scores[:, :, split:], device),
        **weights,
        n_head=4,
        k_cache=TorchTensor.create_from_torch(key.permute(2, 0, 1, 3).reshape(split, -1, 4), device),
        v_cache=TorchTensor.create_from_torch(value.permute(2, 0, 1, 3).reshape(split, -1, 4), device),
        attn_sparsity=1.0,
        rotary_position_ids=positions[:, split:],
    )
    torch.testing.assert_close(torch.cat([first[0].data, second[0].data], dim=1), full[0].data, atol=0.003, rtol=0.003)
    for whole, prefill, following in zip(full[1:], first[1:], second[1:]):
        torch.testing.assert_close(torch.cat([prefill.data, following.data]), whole.data, atol=0.003, rtol=0.003)
