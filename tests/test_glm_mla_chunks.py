"""Chunked GLM prefill must keep the two MLA latent widths intact."""

import pytest
import torch

from bloombee.server import mla_chunk


def _pair(batch, seq, kv_dim, rot_dim, fill):
    kv = torch.full((batch, 1, seq, kv_dim), fill, dtype=torch.float32)
    rot = torch.full((batch, 1, seq, rot_dim), fill + 0.5, dtype=torch.float32)
    return kv, rot


def test_shared_head_view_cannot_represent_glm_flash_latents():
    """The bloom chunk conversion views K and V with K's last dimension."""
    kv_dim, rot_dim = 512, 64  # GLM-4.7-Flash kv_lora_rank, qk_rope_head_dim
    kv, rot = _pair(batch=1, seq=4, kv_dim=kv_dim, rot_dim=rot_dim, fill=1.0)
    bsz, heads, seq, head_dim = kv.shape
    with pytest.raises(RuntimeError):
        rot.reshape(bsz * heads, seq, head_dim)


def test_chunked_prefill_preserves_prefix_and_new_latents():
    # Default server budget is 256MiB. GLM-4.7-Flash has 20 attention heads.
    # A bf16 batch-1 prompt of 4096 tokens costs 20 * 2 * 4096 = 163840 bytes
    # per token of attention scores, so the chunk length is
    # 256MiB // 163840 = 1638 and the prompt is split. This test uses the
    # real latent widths with shorter stand-in chunks.
    kv_dim, rot_dim = 512, 64
    prefix_len, first, second = 5, 4, 3
    prefix = _pair(2, prefix_len, kv_dim, rot_dim, fill=1.0)
    chunk_a = _pair(2, first, kv_dim, rot_dim, fill=2.0)
    chunk_b = _pair(2, second, kv_dim, rot_dim, fill=3.0)

    chunks = []
    layer_past = mla_chunk.record_glm_mla_chunk(prefix, chunk_a, chunks)
    layer_past = mla_chunk.record_glm_mla_chunk(layer_past, chunk_b, chunks)
    written = mla_chunk.concat_mla_chunks(chunks)

    assert layer_past[0].shape == (2, 1, prefix_len + first + second, kv_dim)
    assert layer_past[1].shape == (2, 1, prefix_len + first + second, rot_dim)
    assert written[0].shape == (2, 1, first + second, kv_dim)
    assert written[1].shape == (2, 1, first + second, rot_dim)
    assert torch.equal(layer_past[0][:, :, :prefix_len], prefix[0])
    assert torch.equal(layer_past[1][:, :, prefix_len : prefix_len + first], chunk_a[1])
    assert torch.equal(written[0][:, :, :first], chunk_a[0])
    assert torch.equal(written[0][:, :, first:], chunk_b[0])
    assert torch.equal(written[1][:, :, first:], chunk_b[1])


def test_fresh_prefill_chunk_does_not_require_a_prefix():
    chunk = _pair(1, 3, 512, 64, fill=4.0)
    chunks = []
    layer_past = mla_chunk.record_glm_mla_chunk(None, chunk, chunks)
    written = mla_chunk.concat_mla_chunks(chunks)
    assert torch.equal(layer_past[0], chunk[0])
    assert torch.equal(written[1], chunk[1])


def test_width_mismatch_is_rejected():
    past = _pair(1, 2, 512, 64, fill=1.0)
    new = _pair(1, 2, 256, 64, fill=2.0)
    with pytest.raises(RuntimeError, match="latent widths"):
        mla_chunk.record_glm_mla_chunk(past, new, [])


def test_missing_chunk_cache_is_rejected():
    with pytest.raises(RuntimeError, match="returned no cache"):
        mla_chunk.record_glm_mla_chunk(None, None, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("prefix_len", [0, 3])
async def test_real_backend_chunked_prefill_and_decode_match_hf(batch, prefix_len, make_cpu_cache_manager):
    from transformers import DynamicCache
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRotaryEmbedding,
    )

    from bloombee.data_structures import InferenceMetadata
    from bloombee.models.glm4_moe_lite.block import WrappedGlm4MoeLiteBlock
    from bloombee.models.glm4_moe_lite.config import DistributedGlm4MoeLiteConfig
    from bloombee.server.backend import TransformerBackend
    from bloombee.utils.cache_compat import read_kv_from_cache
    from bloombee.utils.convert_block import QuantType, convert_block
    from bloombee.utils.hivemind_compat import BatchTensorDescriptor

    torch.manual_seed(19)
    config = DistributedGlm4MoeLiteConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=512,
        q_lora_rank=16,
        qk_rope_head_dim=64,
        qk_nope_head_dim=16,
        v_head_dim=16,
        mlp_layer_types=["dense"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    block = WrappedGlm4MoeLiteBlock(config, layer_idx=0).eval()
    reference = Glm4MoeLiteDecoderLayer(config, layer_idx=0).eval()
    reference.load_state_dict(block.state_dict(), strict=False)
    manager = make_cpu_cache_manager(config)
    device = torch.device("cpu")
    module = convert_block(block, 0, config, [device], device, QuantType.NONE)
    descriptor = BatchTensorDescriptor(1, 1, config.hidden_size, dtype=torch.float32)
    length = prefix_len + 7
    backend = TransformerBackend(
        "glm.0",
        module,
        config=config,
        cache_manager=manager,
        pruner_manager=None,
        is_last_block=False,
        backend_dtype=torch.float32,
        block_index=0,
        max_batch_size=1024,
        max_chunk_size_bytes=2 * config.num_attention_heads * batch * 4 * length,
        args_schema=(descriptor,),
        kwargs_schema={},
        outputs_schema=(descriptor,),
    )
    inputs = torch.randn(batch, length + 1, config.hidden_size)
    positions = torch.arange(length + 1).unsqueeze(0).expand(batch, -1)
    mask = torch.triu(torch.full((batch, 1, length + 1, length + 1), float("-inf")), diagonal=1)
    reference_cache = DynamicCache()
    with torch.inference_mode():
        expected = reference(
            inputs,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=Glm4MoeLiteRotaryEmbedding(config)(inputs, positions),
            past_key_values=reference_cache,
            use_cache=True,
        )
        expected = expected[0] if isinstance(expected, tuple) else expected

    # Allocate through the same handle API used by a connection handler.
    manager.runtime_pid += 1
    try:
        async with manager.allocate_cache(*backend.get_inference_cache_descriptors(batch, 16), timeout=0) as handles:

            def step(start, end):
                info = InferenceMetadata(
                    "glm.0", start, tuple(handles), None, full_batch_size=batch, micro_batch_size=batch
                )
                return backend.inference_step(inputs[:, start:end], torch.arange(batch), info)[0]

            if prefix_len:
                torch.testing.assert_close(step(0, prefix_len), expected[:, :prefix_len], atol=1e-5, rtol=1e-5)
            info = InferenceMetadata("glm.0", prefix_len, tuple(handles), None)
            assert backend._estimate_max_chunk_length(inputs[:, prefix_len:length], info) == 2
            actual = step(prefix_len, length)
            torch.testing.assert_close(actual, expected[:, prefix_len:length], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(step(length, length + 1), expected[:, length:], atol=1e-5, rtol=1e-5)
            with manager.use_cache(*handles) as cache_tensors:
                stored = manager.select_mla_cache(length + 1, cache_tensors=cache_tensors)
                for actual_cache, expected_cache in zip(stored, read_kv_from_cache(reference_cache, 0)):
                    assert actual_cache.shape[2] == length + 1
                    torch.testing.assert_close(actual_cache, expected_cache, atol=1e-5, rtol=1e-5)
    finally:
        backend.shutdown()
