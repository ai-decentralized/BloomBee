"""Phase 6 unit test: per-block KV cache allocation honors the TensorDescriptor.

Before Phase 6, ``init_cache_one_gpu_batch`` read head_dim straight off the
global ``config`` (always the sliding-attention value on Gemma-4). When
BloomBee's server tried to write full-attention K/V (head_dim=512) into a
cache allocated with the sliding shape (head_dim=256), the
``memory_cache_manager._write_kvs`` path fired
``assert D_src == D_dst`` and every full-attention block crashed.

Phase 6 adds an optional ``descriptor`` kwarg. When supplied, the cache
shape comes from the descriptor, not the config. This test drives that
contract directly — no GPU, no server, no HF model load.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hivemind.utils import TensorDescriptor

from bloombee.flexgen_utils.pytorch_backend import TorchDevice


def _gemma4_ish_config():
    """Mimic the fields init_cache_one_gpu_batch reads: heterogeneous
    Gemma-4 where ``config.head_dim`` is the SLIDING value (256).
    The bug is that the old code would use this for every layer."""
    return SimpleNamespace(
        num_attention_heads=32,
        hidden_size=5376,
        head_dim=256,          # sliding_attention head_dim
        global_head_dim=512,   # full_attention head_dim (not read directly)
    )


def _task(prompt_len=5, gen_len=4):
    return SimpleNamespace(prompt_len=prompt_len, gen_len=gen_len)


def _policy(batch=1):
    return SimpleNamespace(gpu_batch_size=batch)


def _descriptor(batch, num_heads, head_dim, max_length):
    return TensorDescriptor(size=(batch, num_heads, head_dim, max_length), dtype=torch.float16)


@pytest.fixture
def device():
    return TorchDevice("cpu")  # runs anywhere; we only inspect shapes


def test_sliding_descriptor_matches_config_default(device):
    """Sliding layer descriptor matches config.head_dim=256 — regression check
    that we don't accidentally diverge from legacy behavior."""
    cfg = _gemma4_ish_config()
    task = _task(prompt_len=5, gen_len=4)
    policy = _policy(batch=1)
    descr = _descriptor(batch=1, num_heads=32, head_dim=256, max_length=8)

    k, v = device.init_cache_one_gpu_batch(cfg, task, policy, descriptor=descr)
    # shape is (prompt_len + gen_len - 1, gpu_batch_size * num_heads, head_dim)
    assert k.shape == (8, 32, 256)
    assert v.shape == (8, 32, 256)


def test_full_attention_descriptor_overrides_config_head_dim(device):
    """The bug-fix case: descriptor says head_dim=512, config says 256,
    cache must obey the DESCRIPTOR. Otherwise full-attention block writes
    with D=512 crash against a D=256 cache."""
    cfg = _gemma4_ish_config()
    task = _task(prompt_len=5, gen_len=4)
    policy = _policy(batch=1)
    # Full-attention layer descriptor — matches Gemma4TextAttention output
    # shape when `not is_sliding`: head_dim=global_head_dim=512.
    descr = _descriptor(batch=1, num_heads=32, head_dim=512, max_length=8)

    k, v = device.init_cache_one_gpu_batch(cfg, task, policy, descriptor=descr)
    assert k.shape == (8, 32, 512), (
        f"full-attention cache must honor descriptor head_dim=512, got {k.shape}"
    )
    assert v.shape == (8, 32, 512)


def test_legacy_path_without_descriptor_still_works(device):
    """Uniform families (Llama, Qwen3) don't pass a descriptor — fall back
    to config-driven shape exactly as before. Keeps those paths untouched."""
    cfg = _gemma4_ish_config()
    cfg.head_dim = 128  # say, Llama-7B
    cfg.num_attention_heads = 32
    task = _task(prompt_len=5, gen_len=4)
    policy = _policy(batch=1)

    k, v = device.init_cache_one_gpu_batch(cfg, task, policy)
    assert k.shape == (8, 32, 128)
    assert v.shape == (8, 32, 128)


def test_descriptor_batch_multiplies_into_bh(device):
    """The shape's BH dim equals policy.gpu_batch_size * descriptor.num_heads.
    A batch=4 descriptor with num_heads=32 → BH = 128 rows."""
    cfg = _gemma4_ish_config()
    task = _task(prompt_len=5, gen_len=4)
    policy = _policy(batch=4)
    descr = _descriptor(batch=4, num_heads=32, head_dim=512, max_length=8)

    k, v = device.init_cache_one_gpu_batch(cfg, task, policy, descriptor=descr)
    assert k.shape == (8, 128, 512)


def test_descriptor_vs_config_mismatch_uses_descriptor(device):
    """The whole point of Phase 6: when they disagree, descriptor wins.
    This would have been a silent bug before (config-driven) — now it's
    a property the code enforces."""
    cfg = _gemma4_ish_config()
    task = _task(prompt_len=5, gen_len=4)
    policy = _policy(batch=1)

    # Descriptor with DIFFERENT num_heads + head_dim than config (like
    # Gemma-4 full layer with fewer global KV heads + bigger head_dim).
    descr = _descriptor(batch=1, num_heads=4, head_dim=1024, max_length=8)

    k, v = device.init_cache_one_gpu_batch(cfg, task, policy, descriptor=descr)
    assert k.shape == (8, 4, 1024), f"descriptor should fully override, got {k.shape}"
