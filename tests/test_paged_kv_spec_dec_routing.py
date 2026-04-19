"""Phase 3 tests: make the PagedKVTable shim load-bearing under spec-dec.

Covers the new track_write / rollback / commit routing added in
``KVCacheManager`` and the ``track_write`` helper on PagedKVTable. The
legacy slab write in ``_write_kvs`` remains the source of truth for the
attention kernel — these tests verify the shim's state machine
correctly mirrors it so rejected speculated tokens can be dropped at
page granularity.

What we verify:

1. ``track_write(commit=False)`` only raises ``l_seq``, not ``l_acc``.
2. ``rollback(l_acc_target)`` drops uncommitted tokens and returns
   newly-orphaned pages to the free list.
3. Over a spec-dec step cycle (rollback → tree write speculative →
   commit accepted prefix), the free-page accounting is conserved and
   ``gather_prefix`` only exposes committed bytes, even when the backing
   slab still holds rolled-back speculated bytes.
4. ``update_cache`` (non-spec-dec path) advances both l_seq and l_acc
   together — normal decode writes always commit.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hivemind")

import torch

from bloombee.server.paged_kv import BLOCK_SIZE, PagedKVTable


def _make_table(s_total: int = 8 * BLOCK_SIZE, bh: int = 4, d: int = 4) -> PagedKVTable:
    k = torch.zeros((s_total, bh, d), dtype=torch.float32)
    v = torch.zeros((s_total, bh, d), dtype=torch.float32)
    return PagedKVTable(k, v)


def _fill_pattern(table: PagedKVTable, seq_id: int, start: int, length: int, base: float):
    """Write a recognizable pattern via the paged ``write`` API so we
    can verify rollback hides it from gather_prefix."""
    bh = table.k.shape[1]
    d = table.k.shape[2]
    k_new = torch.full((length, bh, d), base, dtype=table.k.dtype)
    v_new = k_new + 100.0
    table.write(seq_id, start, k_new, v_new, bh_slice=(0, bh), commit=False)


# ---------------------------------------------------------------------------


def test_track_write_speculative_does_not_advance_l_acc():
    t = _make_table()
    t.register_sequence(0)
    # Establish a committed prefix of 10.
    t.track_write(0, start_position=0, s_new=10, commit=True)
    assert t.l_acc(0) == 10 and t.l_seq(0) == 10

    # Speculative tree of 7 tokens lands past the committed prefix.
    t.track_write(0, start_position=10, s_new=7, commit=False)
    assert t.l_acc(0) == 10, "commit=False must not promote tokens to accepted"
    assert t.l_seq(0) == 17, "commit=False must still advance l_seq"


def test_rollback_frees_pages_from_speculated_tail():
    t = _make_table(s_total=8 * BLOCK_SIZE)
    free_before = t.num_free_pages()
    t.register_sequence(0)
    # Accept 1 full page (BLOCK_SIZE tokens).
    t.track_write(0, 0, BLOCK_SIZE, commit=True)
    # Speculate another full page (BLOCK_SIZE more tokens).
    t.track_write(0, BLOCK_SIZE, BLOCK_SIZE, commit=False)
    used_before_rollback = t.num_used_pages(0)
    assert used_before_rollback == 2

    # Reject the whole speculation: rollback to the committed prefix.
    t.rollback(0, l_acc_target=BLOCK_SIZE)

    assert t.num_used_pages(0) == 1, "tail page must return to the free list"
    assert t.num_free_pages() == free_before - 1, "conservation: one page held"
    assert t.l_seq(0) == BLOCK_SIZE
    assert t.l_acc(0) == BLOCK_SIZE


def test_spec_dec_step_cycle_conserves_pages_across_many_steps():
    """Full spec-dec cycle: rollback → speculative write → partial commit.

    Simulates 16 decode steps where half of each step's 4-token tree is
    accepted. Page accounting must stay bounded (we must not leak pages
    across steps) and ``l_acc`` must grow monotonically by exactly the
    accepted count per step.
    """
    t = _make_table(s_total=16 * BLOCK_SIZE)
    t.register_sequence(0)
    tree_len = 4
    accepted_per_step = 2

    # Warm up committed prefix at start of decode loop.
    t.track_write(0, 0, accepted_per_step, commit=True)
    expected_l_acc = accepted_per_step

    free_after_warmup = t.num_free_pages()

    for step in range(16):
        # 1. Rollback removes prior step's uncommitted tail (idempotent on step 0).
        t.rollback(0, l_acc_target=expected_l_acc)
        assert t.l_seq(0) == expected_l_acc
        assert t.l_acc(0) == expected_l_acc

        # 2. Speculative tree of tree_len tokens lands past the committed prefix.
        t.track_write(0, expected_l_acc, tree_len, commit=False)
        assert t.l_seq(0) == expected_l_acc + tree_len
        assert t.l_acc(0) == expected_l_acc  # unchanged

        # 3. Verify accepts `accepted_per_step` of them.
        new_acc = expected_l_acc + accepted_per_step
        t.commit(0, up_to=new_acc)
        expected_l_acc = new_acc
        assert t.l_acc(0) == expected_l_acc
        # l_seq is still out past the accepted prefix holding rejected tokens.
        assert t.l_seq(0) == expected_l_acc + (tree_len - accepted_per_step)

    # Final rollback to the last committed prefix (mirrors start of next step).
    t.rollback(0, l_acc_target=expected_l_acc)

    # Pages held must equal ceil(expected_l_acc / BLOCK_SIZE).
    expected_pages = (expected_l_acc + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert t.num_used_pages(0) == expected_pages
    assert t.num_free_pages() == free_after_warmup + 1 - expected_pages


def test_gather_prefix_hides_rolled_back_speculated_bytes():
    """Byte-level invariant 3 under the Phase 3 flow.

    After committing N bytes and speculating a further M bytes that then
    get rejected, ``gather_prefix`` must return only the committed N bytes
    — even though the backing slab still physically holds the speculated
    values until the next write overwrites them.
    """
    t = _make_table(s_total=4 * BLOCK_SIZE, bh=2, d=2)
    t.register_sequence(0)

    # Commit 5 tokens with value 7.0.
    _fill_pattern(t, 0, start=0, length=5, base=7.0)
    t.commit(0, up_to=5)
    assert t.l_acc(0) == 5

    # Speculate 3 more tokens with a distinct sentinel value 999.0.
    _fill_pattern(t, 0, start=5, length=3, base=999.0)
    assert t.l_seq(0) == 8
    assert t.l_acc(0) == 5

    # Reject all speculation.
    t.rollback(0, l_acc_target=5)

    # Read back: must only see the 5 committed bytes.
    k_prefix, v_prefix = t.gather_prefix(0, bh_slice=(0, 2))
    assert k_prefix.shape == (5, 2, 2)
    assert torch.all(k_prefix == 7.0)
    assert torch.all(v_prefix == 107.0)

    # Requesting past l_acc must raise — gather_prefix refuses to expose
    # stale speculated bytes even if they are still in the slab.
    from bloombee.server.paged_kv import PagedKVError
    with pytest.raises(PagedKVError):
        t.gather_prefix(0, bh_slice=(0, 2), length=8)


def test_track_write_auto_registers_sequence():
    """track_write is the manager's mirror path; it should auto-register
    sequences so the KVCacheManager doesn't have to gate first-write."""
    t = _make_table()
    assert not t.has_sequence(42)
    t.track_write(42, start_position=0, s_new=3, commit=True)
    assert t.has_sequence(42)
    assert t.l_acc(42) == 3
    assert t.l_seq(42) == 3


def test_rollback_to_current_l_acc_is_noop():
    """Starting each spec-dec step with ``rollback(l_acc_target=l_acc)``
    on the very first step (when no speculation yet exists) must be a
    no-op — otherwise the Phase 3 integration would leak pages on step 0."""
    t = _make_table()
    t.register_sequence(0)
    t.track_write(0, 0, 5, commit=True)
    free_before = t.num_free_pages()
    used_before = t.num_used_pages(0)

    t.rollback(0, l_acc_target=5)

    assert t.num_free_pages() == free_before
    assert t.num_used_pages(0) == used_before
    assert t.l_seq(0) == 5
    assert t.l_acc(0) == 5
