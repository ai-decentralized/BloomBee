"""Unit test for TransformerBackend's per-layer head_dim dispatch.

Phase 2 introduces `block_index` + `_head_dim_for_this_block` so that
Gemma-4's sliding layers and full layers end up with the right
(num_heads, head_dim) cache shape. This test stubs out the inherited
ModuleBackend pieces and drives _head_dim_for_this_block directly, so it
runs on CPU without torch/hivemind init.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bloombee.server.backend import TransformerBackend


def _stub_backend(config, block_index):
    """Build a barely-valid TransformerBackend without calling __init__."""
    b = TransformerBackend.__new__(TransformerBackend)
    b.config = config
    b.block_index = block_index
    return b


def _gemma4_like_config():
    """Tiny config mirroring Gemma-4-31B's relevant fields."""
    return SimpleNamespace(
        hidden_size=5376,
        num_attention_heads=32,
        head_dim=256,              # sliding_attention
        global_head_dim=512,       # full_attention
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        layer_types=(
            ["sliding_attention"] * 5 + ["full_attention"]
            + ["sliding_attention"] * 5 + ["full_attention"]
        ),
    )


def _qwen3_like_config():
    """Uniform-attention config — must use default head_dim regardless of block_index."""
    return SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=40,
        head_dim=128,
    )


def test_gemma4_sliding_layers_use_head_dim():
    cfg = _gemma4_like_config()
    # indices 0..4 are sliding on the mini pattern
    for bi in (0, 1, 2, 3, 4):
        b = _stub_backend(cfg, bi)
        assert b._layer_type_for_this_block() == "sliding_attention"
        assert b._head_dim_for_this_block() == 256


def test_gemma4_full_layers_use_global_head_dim():
    cfg = _gemma4_like_config()
    # index 5 is full_attention (the "every 6th" layer)
    for bi in (5, 11):
        b = _stub_backend(cfg, bi)
        assert b._layer_type_for_this_block() == "full_attention"
        assert b._head_dim_for_this_block() == 512


def test_uniform_family_falls_back_to_default_head_dim():
    cfg = _qwen3_like_config()
    # No layer_types attr → layer_type is None → default head_dim.
    for bi in (None, 0, 5, 63):
        b = _stub_backend(cfg, bi)
        assert b._layer_type_for_this_block() is None
        assert b._head_dim_for_this_block() == 128


def test_missing_block_index_falls_back_to_default():
    """Backends constructed before Phase 2 (no block_index kwarg) must
    continue to work on uniform families by defaulting to head_dim."""
    cfg = _gemma4_like_config()
    b = _stub_backend(cfg, None)
    assert b._layer_type_for_this_block() is None
    assert b._head_dim_for_this_block() == 256  # default head_dim, not global_head_dim


def test_out_of_range_block_index_is_robust():
    """Defensive: a negative or past-the-end block_index shouldn't crash."""
    cfg = _gemma4_like_config()
    for bi in (-1, 99):
        b = _stub_backend(cfg, bi)
        assert b._layer_type_for_this_block() is None
        assert b._head_dim_for_this_block() == 256
