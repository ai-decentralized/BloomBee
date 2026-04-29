"""Phase 2 substrate-compatibility test.

The Phase 2 hot-path shim replaces ``MemoryCache._allocated_tensors[handle]``
with a PagedKVTable whose backing store is the same ``(S, BH, D)`` slab that
``init_cache_one_gpu_batch`` already produces. For the swap to be
transparent, two invariants must hold:

1. A sequence that writes a contiguous prefix via ``table.write`` must
   leave the slab in byte-identical state to the legacy
   ``k[pos : pos + s_new] = k_new`` path used by ``mha_gen_llama``.
2. Spec-decoded writes past ``l_acc`` that are rolled back must NOT leak
   bytes through ``gather_prefix``; the committed read path must stay
   clean even when the underlying slab still holds the speculated tokens.

This runs as a plain unit test (no server, no CUDA) — it operates
directly on CPU tensors in the same layout FlexGen uses internally.
"""

from __future__ import annotations

import pytest
import torch

from bloombee.server.paged_kv import BLOCK_SIZE, PagedKVError, PagedKVTable


def _make_slab_pair(num_pages: int, bh: int, d: int, dtype: torch.dtype = torch.float32):
    s_total = num_pages * BLOCK_SIZE
    k = torch.zeros((s_total, bh, d), dtype=dtype)
    v = torch.zeros((s_total, bh, d), dtype=dtype)
    return k, v


# -------------------------------------------------------------------- #
# Invariant 1: paged writes reproduce the contiguous slab bytes that    #
# mha_gen_llama produces when its in-place branch fires.                #
# -------------------------------------------------------------------- #

@pytest.mark.parametrize("s_new,start", [
    (1, 0),
    (1, 31),               # straddles page boundary 1→2
    (7, 9),                # intra-page
    (BLOCK_SIZE, 0),       # exactly one page
    (BLOCK_SIZE + 3, 14),  # crosses 2 pages
    (2 * BLOCK_SIZE + 1, 0),  # crosses 3 pages
])
def test_paged_write_matches_inplace_slab_write(s_new, start):
    """The exact rewrite that the Phase 2 shim performs must be bit-identical
    to the legacy slab write path from mha_gen_llama."""
    bh, d = 8, 4
    num_pages = 8

    # Reference path — what mha_gen_llama's in-place branch would do today.
    ref_k, ref_v = _make_slab_pair(num_pages, bh, d)
    k_new = torch.randn(s_new, bh, d)
    v_new = torch.randn(s_new, bh, d) + 100.0
    ref_k[start : start + s_new] = k_new
    ref_v[start : start + s_new] = v_new

    # Paged substrate path — Phase 2 routes writes through PagedKVTable.
    shim_k, shim_v = _make_slab_pair(num_pages, bh, d)
    table = PagedKVTable(shim_k, shim_v)
    table.register_sequence(0)
    if start > 0:
        # Warm-start at non-zero position: initialise l_seq/l_acc so the
        # paged write lands where the slab write would. The shim has to
        # establish the prior prefix's page mapping before accepting the
        # next write.
        pre_k = torch.zeros(start, bh, d)
        pre_v = torch.zeros(start, bh, d)
        table.write(0, 0, pre_k, pre_v, bh_slice=(0, bh))

    table.write(0, start, k_new, v_new, bh_slice=(0, bh))

    # Sequential page allocation guarantees pages [0..N-1] live at
    # contiguous slab offsets, so shim_k/shim_v must equal ref_k/ref_v.
    torch.testing.assert_close(shim_k, ref_k)
    torch.testing.assert_close(shim_v, ref_v)

    # gather_prefix from the paged table yields the same bytes the
    # attention kernel would read off the slab as [: l_acc].
    l_acc = start + s_new
    gathered_k, gathered_v = table.gather_prefix(0, bh_slice=(0, bh))
    torch.testing.assert_close(gathered_k, ref_k[:l_acc])
    torch.testing.assert_close(gathered_v, ref_v[:l_acc])


# -------------------------------------------------------------------- #
# Invariant 2: rollback never leaks speculated bytes through the read   #
# API even though those bytes are still resident in the slab.           #
# -------------------------------------------------------------------- #

def test_rollback_hides_speculated_bytes_from_gather_prefix():
    bh, d = 4, 3
    num_pages = 6
    k, v = _make_slab_pair(num_pages, bh, d)
    table = PagedKVTable(k, v)
    table.register_sequence(42)

    # Committed prefix (L_acc = 20).
    commit_k = torch.randn(20, bh, d)
    commit_v = torch.randn(20, bh, d)
    table.write(42, 0, commit_k, commit_v, bh_slice=(0, bh))
    assert table.l_acc(42) == 20
    assert table.l_seq(42) == 20

    # Speculative tree extension: write 12 more tokens past L_acc, don't
    # commit. These bytes are definitely present in the slab storage.
    spec_k = torch.randn(12, bh, d) * 777.0  # distinctive magnitude
    spec_v = torch.randn(12, bh, d) * 777.0
    table.write(42, 20, spec_k, spec_v, bh_slice=(0, bh), commit=False)
    assert table.l_acc(42) == 20
    assert table.l_seq(42) == 32

    # Sanity: the speculated bytes really are in the backing slab.
    slab_prefix = k[:32].clone()
    assert not torch.allclose(slab_prefix[20:32], torch.zeros_like(slab_prefix[20:32]))

    # Rollback — no acceptance.
    table.rollback(42, l_acc_target=20)
    assert table.l_acc(42) == 20
    assert table.l_seq(42) == 20

    # Read the committed prefix. Even though the slab storage still
    # contains the speculated bytes, gather_prefix must return only the
    # accepted prefix — this is invariant 3 from PHASE2 docs.
    gathered_k, gathered_v = table.gather_prefix(42, bh_slice=(0, bh))
    assert gathered_k.shape == (20, bh, d)
    torch.testing.assert_close(gathered_k, commit_k)
    torch.testing.assert_close(gathered_v, commit_v)

    # Requesting a length past L_acc must raise: contract says the
    # shim cannot be coerced into exposing speculated bytes.
    with pytest.raises(PagedKVError):
        table.gather_prefix(42, bh_slice=(0, bh), length=32)


# -------------------------------------------------------------------- #
# Invariant 3: pages freed by rollback are actually reusable; the shim  #
# cannot leak pages on each rolled-back speculation.                    #
# -------------------------------------------------------------------- #

def test_rollback_returns_pages_to_free_list():
    bh, d = 2, 2
    num_pages = 4
    k, v = _make_slab_pair(num_pages, bh, d)
    table = PagedKVTable(k, v)
    table.register_sequence(0)

    # Start with one page committed.
    table.write(0, 0, torch.randn(BLOCK_SIZE, bh, d), torch.randn(BLOCK_SIZE, bh, d),
                bh_slice=(0, bh))
    assert table.num_used_pages(0) == 1
    assert table.num_free_pages() == num_pages - 1

    # Speculate into pages 2 and 3.
    spec = torch.randn(2 * BLOCK_SIZE, bh, d)
    table.write(0, BLOCK_SIZE, spec, spec, bh_slice=(0, bh), commit=False)
    assert table.num_used_pages(0) == 3
    assert table.num_free_pages() == num_pages - 3

    # Roll back — the two speculated pages should be returned to the pool.
    table.rollback(0, l_acc_target=BLOCK_SIZE)
    assert table.num_used_pages(0) == 1
    assert table.num_free_pages() == num_pages - 1

    # And they must be re-allocatable by a second sequence — proves the
    # shim won't deadlock a decode loop that speculates-rejects-repeats.
    table.register_sequence(1)
    table.write(1, 0, torch.randn(2 * BLOCK_SIZE, bh, d),
                torch.randn(2 * BLOCK_SIZE, bh, d), bh_slice=(0, bh))
    assert table.num_used_pages(1) == 2
    # 4 total pages: 1 (seq 0 committed) + 2 (seq 1) + 1 free.
    assert table.num_free_pages() == 1
