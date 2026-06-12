"""WrappedBloomBlock must hand BloomBee only the NEW tokens' KV.

update_cache() writes `present` at start_position, so a present that contains
the full past+new concatenation corrupts the cache write on every decode step
(observed live as `general_copy` size mismatches on bloom-560m)."""

import pytest
import torch
from transformers.models.bloom.modeling_bloom import BloomConfig

from bloombee.models.bloom.block import WrappedBloomBlock


@pytest.fixture
def tiny_block():
    config = BloomConfig(hidden_size=64, n_head=4, n_layer=2)
    block = WrappedBloomBlock(config, layer_idx=0)
    block.eval()
    return block, config


def _bh(config):
    return config.n_head  # batch=1 below, so B*H == n_head


def test_prefill_present_covers_prompt(tiny_block):
    block, config = tiny_block
    prompt_len = 4
    hidden = torch.randn(1, prompt_len, config.hidden_size)
    with torch.no_grad():
        _, present = block(hidden, use_cache=True)
    k, v = present
    head_dim = config.hidden_size // config.n_head
    assert k.shape == (_bh(config), head_dim, prompt_len)
    assert v.shape == (_bh(config), prompt_len, head_dim)


def test_decode_present_contains_only_new_tokens(tiny_block):
    block, config = tiny_block
    prompt_len = 4
    head_dim = config.hidden_size // config.n_head
    hidden = torch.randn(1, prompt_len, config.hidden_size)
    with torch.no_grad():
        _, prefill_present = block(hidden, use_cache=True)
        next_token = torch.randn(1, 1, config.hidden_size)
        # backend hands the cache back in the same [B*H, D, S] / [B*H, S, D] layout
        _, decode_present = block(next_token, layer_past=prefill_present, use_cache=True)
    k, v = decode_present
    assert k.shape == (_bh(config), head_dim, 1), f"present must be new-tokens-only, got k {k.shape}"
    assert v.shape == (_bh(config), 1, head_dim), f"present must be new-tokens-only, got v {v.shape}"
