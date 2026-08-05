#!/usr/bin/env python3
"""Unit tests for active-row compaction primitives (no GPU/swarm needed).

1. KV slab front-gather semantics: (S, B*H, D) slab, gather rows [perm] to front,
   verify gathered rows byte-identical to their sources and untouched rows beyond
   n stay dead (never read).
2. _cap_valid_lengths_to_remaining with per-row (tensor) max_new_tokens.
3. EAGLEDrafter.reorder_prefix_states remap logic (pure dict, no model load).
"""
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def test_slab_front_gather():
    S, B, H, D = 7, 6, 4, 5
    torch.manual_seed(0)
    k = torch.randn(S, B * H, D)
    v = torch.randn(S, B * H, D)
    k_orig, v_orig = k.clone(), v.clone()
    perm = torch.tensor([0, 2, 5])  # rows 1,3,4 finished
    n = perm.numel()
    bh_perm = (perm * H).repeat_interleave(H) + torch.arange(H).repeat(n)
    k[:, : n * H, :] = k[:, bh_perm, :]
    v[:, : n * H, :] = v[:, bh_perm, :]
    for new_i, old_i in enumerate(perm.tolist()):
        assert torch.equal(k[:, new_i * H : (new_i + 1) * H, :], k_orig[:, old_i * H : (old_i + 1) * H, :]), f"k row {new_i}"
        assert torch.equal(v[:, new_i * H : (new_i + 1) * H, :], v_orig[:, old_i * H : (old_i + 1) * H, :]), f"v row {new_i}"
    # overlapping gather (src row 2 lands in dst row 1's slot) must still be exact
    perm2 = torch.tensor([1, 2, 3, 4])
    k2 = k_orig.clone()
    bh2 = (perm2 * H).repeat_interleave(H) + torch.arange(H).repeat(perm2.numel())
    k2[:, : perm2.numel() * H, :] = k2[:, bh2, :]
    for new_i, old_i in enumerate(perm2.tolist()):
        assert torch.equal(k2[:, new_i * H : (new_i + 1) * H, :], k_orig[:, old_i * H : (old_i + 1) * H, :]), f"overlap row {new_i}"
    print("PASS slab front-gather (incl. overlapping src/dst)")


def test_cap_valid_lengths_per_row():
    from bloombee.models.llama.speculative_model import _cap_valid_lengths_to_remaining

    seq = torch.tensor([10, 20, 30])
    init = torch.tensor([5, 5, 5])
    valid = torch.tensor([4, 4, 4])
    quotas = torch.tensor([6, 16, 100])  # remaining: 1, 1, 75
    capped, append = _cap_valid_lengths_to_remaining(valid, seq, init, quotas)
    assert capped.tolist() == [1, 1, 4], capped
    assert append.tolist() == [0, 0, 1], append
    # scalar path unchanged
    capped_s, append_s = _cap_valid_lengths_to_remaining(valid, seq, init, 100)
    assert capped_s.tolist() == [4, 4, 4] and append_s.tolist() == [1, 1, 1]
    print("PASS per-row quota capping")


def test_reorder_prefix_states():
    from bloombee.models.llama.eagle_drafter import EAGLEDrafter

    class _Stub:
        _prefix_states = {0: "s0", 1: "s1", 2: "s2", 3: "s3"}
        reorder_prefix_states = EAGLEDrafter.reorder_prefix_states

    stub = _Stub()
    stub.reorder_prefix_states([1, 3])  # rows 0,2 finished; survivors 1,3
    assert stub._prefix_states == {0: "s1", 1: "s3"}, stub._prefix_states
    stub.reorder_prefix_states([0])
    assert stub._prefix_states == {0: "s1"}
    print("PASS drafter prefix-state remap")


if __name__ == "__main__":
    test_slab_front_gather()
    test_cap_valid_lengths_per_row()
    test_reorder_prefix_states()
    print("ALL UNIT TESTS PASS")
