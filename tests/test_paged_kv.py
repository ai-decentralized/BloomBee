"""Unit tests for PagedKVTable (Phase 2)."""

import pytest
import torch

from bloombee.server.paged_kv import BLOCK_SIZE, PagedKVError, PagedKVTable


def _make_table(num_pages=4, bh=2, d=3, dtype=torch.float32):
    s_total = num_pages * BLOCK_SIZE
    k = torch.zeros((s_total, bh, d), dtype=dtype)
    v = torch.zeros((s_total, bh, d), dtype=dtype)
    return PagedKVTable(k, v), k, v


def test_init_requires_multiple_of_block_size():
    k = torch.zeros((10, 2, 3))
    v = torch.zeros((10, 2, 3))
    with pytest.raises(PagedKVError):
        PagedKVTable(k, v)


def test_register_and_release_returns_pages_to_free_list():
    table, _, _ = _make_table(num_pages=4)
    table.register_sequence(0)
    k = torch.randn((BLOCK_SIZE + 5, 2, 3))
    v = torch.randn((BLOCK_SIZE + 5, 2, 3))
    table.write(0, 0, k, v, bh_slice=(0, 2))
    assert table.num_used_pages(0) == 2
    assert table.num_free_pages() == 2
    table.release_sequence(0)
    assert table.num_free_pages() == 4


def test_basic_write_and_read_roundtrip():
    table, _, _ = _make_table(num_pages=4)
    table.register_sequence(7)
    s_new = BLOCK_SIZE + 5  # straddle page boundary
    k_src = torch.arange(s_new * 2 * 3, dtype=torch.float32).reshape(s_new, 2, 3)
    v_src = k_src + 100.0
    table.write(7, 0, k_src, v_src, bh_slice=(0, 2))
    assert table.l_seq(7) == s_new
    assert table.l_acc(7) == s_new
    k_out, v_out = table.gather_prefix(7, bh_slice=(0, 2))
    torch.testing.assert_close(k_out, k_src)
    torch.testing.assert_close(v_out, v_src)


def test_speculative_write_then_rollback_preserves_accepted_prefix():
    table, _, _ = _make_table(num_pages=4)
    table.register_sequence(0)

    accepted = torch.full((BLOCK_SIZE,), 1.0).view(BLOCK_SIZE, 1, 1).expand(BLOCK_SIZE, 2, 3).contiguous()
    table.write(0, 0, accepted, accepted + 10, bh_slice=(0, 2), commit=True)
    assert table.l_acc(0) == BLOCK_SIZE
    assert table.num_used_pages(0) == 1

    # Spec writes that will be rolled back (straddle into a second page).
    spec = torch.full((10,), 2.0).view(10, 1, 1).expand(10, 2, 3).contiguous()
    table.write(0, BLOCK_SIZE, spec, spec + 10, bh_slice=(0, 2), commit=False)
    assert table.l_seq(0) == BLOCK_SIZE + 10
    assert table.l_acc(0) == BLOCK_SIZE
    assert table.num_used_pages(0) == 2

    # Rollback drops the second page + marks speculation dropped.
    table.rollback(0, BLOCK_SIZE)
    assert table.l_seq(0) == BLOCK_SIZE
    assert table.l_acc(0) == BLOCK_SIZE
    assert table.num_used_pages(0) == 1

    # Reading returns accepted prefix only; spec bytes are gone with the page.
    k_out, _ = table.gather_prefix(0, bh_slice=(0, 2))
    torch.testing.assert_close(k_out, accepted)


def test_rollback_inside_a_page_keeps_the_page():
    """Partial rollback within the last page must not release it (invariant 3)."""
    table, _, _ = _make_table(num_pages=4)
    table.register_sequence(0)
    committed = torch.full((5,), 1.0).view(5, 1, 1).expand(5, 2, 3).contiguous()
    table.write(0, 0, committed, committed, bh_slice=(0, 2), commit=True)
    spec = torch.full((3,), 9.0).view(3, 1, 1).expand(3, 2, 3).contiguous()
    table.write(0, 5, spec, spec, bh_slice=(0, 2), commit=False)
    assert table.l_seq(0) == 8
    assert table.num_used_pages(0) == 1

    table.rollback(0, 5)
    assert table.num_used_pages(0) == 1
    assert table.num_free_pages() == 3

    k_out, _ = table.gather_prefix(0, bh_slice=(0, 2))
    assert k_out.shape[0] == 5
    torch.testing.assert_close(k_out, committed)


def test_read_is_clamped_to_l_acc():
    table, _, _ = _make_table(num_pages=4)
    table.register_sequence(0)
    ones = torch.full((4,), 1.0).view(4, 1, 1).expand(4, 2, 3).contiguous()
    table.write(0, 0, ones, ones, bh_slice=(0, 2), commit=True)
    twos = torch.full((2,), 2.0).view(2, 1, 1).expand(2, 2, 3).contiguous()
    table.write(0, 4, twos, twos, bh_slice=(0, 2), commit=False)
    # Asking for l_acc returns 4 ones; asking past l_acc raises.
    k_out, _ = table.gather_prefix(0, bh_slice=(0, 2))
    assert k_out.shape[0] == 4
    torch.testing.assert_close(k_out, ones)
    with pytest.raises(PagedKVError):
        table.gather_prefix(0, bh_slice=(0, 2), length=5)


def test_write_idempotence_within_unaccepted_region():
    """Rewriting [l_acc, l_seq) must not corrupt the committed prefix (invariant 4)."""
    table, _, _ = _make_table(num_pages=4)
    table.register_sequence(0)
    committed = torch.full((4,), 1.0).view(4, 1, 1).expand(4, 2, 3).contiguous()
    table.write(0, 0, committed, committed, bh_slice=(0, 2), commit=True)
    spec_a = torch.full((3,), 7.0).view(3, 1, 1).expand(3, 2, 3).contiguous()
    table.write(0, 4, spec_a, spec_a, bh_slice=(0, 2), commit=False)
    # Redo the spec write with a different value — must work.
    spec_b = torch.full((3,), 8.0).view(3, 1, 1).expand(3, 2, 3).contiguous()
    table.write(0, 4, spec_b, spec_b, bh_slice=(0, 2), commit=False)
    # Committed bytes unchanged.
    k_out, _ = table.gather_prefix(0, bh_slice=(0, 2), length=4)
    torch.testing.assert_close(k_out, committed)


def test_out_of_pages_raises():
    table, _, _ = _make_table(num_pages=2)  # total 32 tokens
    table.register_sequence(0)
    too_many = torch.zeros((40, 2, 3))
    with pytest.raises(PagedKVError):
        table.write(0, 0, too_many, too_many, bh_slice=(0, 2))


def test_bh_slice_respected_microbatch():
    """Writes into a BH slice must not disturb the rest of the backing tensor."""
    table, k_backing, v_backing = _make_table(num_pages=4, bh=4, d=3)
    table.register_sequence(0)
    sentinel = torch.full_like(k_backing, -1.0)
    k_backing.copy_(sentinel)
    v_backing.copy_(sentinel)

    payload = torch.ones((5, 2, 3))
    table.write(0, 0, payload, payload + 1, bh_slice=(1, 3))

    # BH slice [1:3] over the written range should be 1.0 / 2.0; the rest untouched.
    torch.testing.assert_close(k_backing[:5, 1:3, :], torch.ones((5, 2, 3)))
    torch.testing.assert_close(v_backing[:5, 1:3, :], torch.full((5, 2, 3), 2.0))
    assert (k_backing[:5, 0, :] == -1.0).all()
    assert (k_backing[:5, 3, :] == -1.0).all()


def test_commit_bumps_l_acc():
    table, _, _ = _make_table(num_pages=2)
    table.register_sequence(0)
    payload = torch.ones((5, 2, 3))
    table.write(0, 0, payload, payload, bh_slice=(0, 2), commit=False)
    assert table.l_acc(0) == 0
    table.commit(0)
    assert table.l_acc(0) == 5
    # Partial commit is rejected if beyond l_seq.
    with pytest.raises(PagedKVError):
        table.commit(0, up_to=10)
