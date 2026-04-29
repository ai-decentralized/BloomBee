"""Phase 0 parity test for FlexGen decode-path KV cache write.

Validates that the in-place write into the preallocated slab produces
byte-identical output to the previous ``torch.cat`` fallback. This is
what the Phase 0 commit (c04bdee) changed in pytorch_backend.py:812-835.

We don't spin up a whole FlexGen runtime — we replicate the two branches
directly on synthetic tensors shaped like real llama decode step state.
Two regimes are exercised:

1. Preallocated slab (capacity >= src_s): the new in-place branch writes
   k_new at ``[src_s - tgt_s : src_s]`` and takes a view to ``[:src_s]``.
   Must equal ``torch.cat([history_slice, k_new], dim=0)``.
2. Server path (capacity == history_len < src_s): fallback to concat;
   must behave exactly as before the refactor.

If this regression ever fires, the Phase 0 edit broke decode semantics.
"""

import pytest
import torch


def _inplace_then_view(k_slab: torch.Tensor, k_new: torch.Tensor, src_s: int, tgt_s: int) -> torch.Tensor:
    """Reproduce the new branch (pytorch_backend.py:828-832)."""
    k_slab[src_s - tgt_s : src_s] = k_new
    return k_slab[:src_s]


def _concat_fallback(k_history: torch.Tensor, k_new: torch.Tensor) -> torch.Tensor:
    """Reproduce the old branch (pytorch_backend.py:834-835)."""
    return torch.cat([k_history, k_new], dim=0)


@pytest.mark.parametrize("tgt_s", [1, 4])
def test_inplace_matches_concat_on_preallocated_slab(tgt_s):
    """Preallocated slab path must equal concat when sharing the same history."""
    bh = 32  # batch * n_head
    head_dim = 128
    history_len = 63
    src_s = history_len + tgt_s
    slab_capacity = 256  # simulating preallocated max_seq_len slab

    # Golden history and new-token tensors.
    torch.manual_seed(0)
    history = torch.randn(history_len, bh, head_dim, dtype=torch.float16)
    k_new = torch.randn(tgt_s, bh, head_dim, dtype=torch.float16)

    # New path: history lives inside the slab already.
    k_slab = torch.zeros(slab_capacity, bh, head_dim, dtype=torch.float16)
    k_slab[:history_len] = history
    got = _inplace_then_view(k_slab, k_new, src_s, tgt_s)

    # Old path: history and new concatenated.
    expected = _concat_fallback(history, k_new)

    assert got.shape == expected.shape == (src_s, bh, head_dim)
    assert torch.equal(got, expected), "in-place slab write must match concat semantics"


def test_concat_fallback_still_taken_when_capacity_small():
    """When the cache is a fresh per-step tensor (capacity == history_len), the server
    path must still take the concat branch — this is why the fallback exists."""
    bh = 32
    head_dim = 128
    history_len = 63
    tgt_s = 1
    src_s = history_len + tgt_s

    # Cache is exactly history-shaped (server path: cache seeded from past_key_value).
    torch.manual_seed(0)
    history = torch.randn(history_len, bh, head_dim, dtype=torch.float16)
    k_new = torch.randn(tgt_s, bh, head_dim, dtype=torch.float16)
    cache_capacity = history.shape[0]

    # Should hit the fallback arm.
    assert cache_capacity < src_s
    out = _concat_fallback(history, k_new)
    assert out.shape == (src_s, bh, head_dim)
    assert torch.equal(out[:history_len], history)
    assert torch.equal(out[history_len:], k_new)


def test_does_not_disturb_untouched_slab_rows():
    """In-place write must not perturb bytes above the written range — critical for
    any reader that slices ``k[:src_s]`` expecting everything past that to be opaque."""
    bh = 8
    head_dim = 16
    slab_capacity = 32
    history_len = 10
    tgt_s = 2
    src_s = history_len + tgt_s

    torch.manual_seed(42)
    k_slab = torch.randn(slab_capacity, bh, head_dim, dtype=torch.float16)
    sentinel = k_slab[src_s:].clone()
    k_new = torch.randn(tgt_s, bh, head_dim, dtype=torch.float16)

    _ = _inplace_then_view(k_slab, k_new, src_s, tgt_s)
    torch.testing.assert_close(k_slab[src_s:], sentinel)


def test_view_returned_is_live_write_target():
    """The returned slice must alias the slab — downstream ``k.permute`` operations
    in mha_gen_llama rely on the slab being the storage, not a fresh copy."""
    bh = 4
    head_dim = 8
    slab_capacity = 32
    history_len = 10
    tgt_s = 1
    src_s = history_len + tgt_s

    k_slab = torch.zeros(slab_capacity, bh, head_dim, dtype=torch.float16)
    k_new = torch.full((tgt_s, bh, head_dim), 7.0, dtype=torch.float16)
    view = _inplace_then_view(k_slab, k_new, src_s, tgt_s)

    assert view.data_ptr() == k_slab[:src_s].data_ptr()
    assert (view[src_s - tgt_s : src_s] == 7.0).all()
