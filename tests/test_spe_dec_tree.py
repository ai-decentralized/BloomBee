"""Regression tests for spe_dec_tree ancestor matrix optimization.

Confirms the O(n*depth) parent-walk implementation is byte-identical to
the previous O(n^3) matmul-based transitive closure on the realistic tree
shapes BloomBee uses (depth-3/4 speculation trees up to width 8).
"""

import torch

from bloombee.models.llama.spe_dec_tree import (
    build_ancestor_matrix_optimized,
    build_incremental_tree_attention_mask,
)


def _reference_ancestor_matrix(parent_indices, device):
    """Old matmul-based transitive closure, kept for byte-exact regression."""
    n = len(parent_indices)
    if n == 0:
        return torch.empty(0, 0, dtype=torch.bool, device=device)
    A = torch.zeros(n, n, dtype=torch.bool, device=device)
    rows = torch.arange(n, device=device)
    cols = torch.as_tensor(parent_indices, device=device)
    mask = cols >= 0
    if mask.any():
        A[rows[mask], cols[mask]] = True
    ancestor_matrix = A.clone()
    for _ in range(n):
        A_f = A.float()
        anc_f = ancestor_matrix.float()
        power = torch.matmul(anc_f, A_f)
        new_reach = ancestor_matrix | (power > 0)
        if torch.equal(new_reach, ancestor_matrix):
            break
        ancestor_matrix = new_reach
    return ancestor_matrix


def _check(parents):
    device = torch.device("cpu")
    new = build_ancestor_matrix_optimized(parents, device)
    old = _reference_ancestor_matrix(parents, device)
    assert torch.equal(new, old), f"mismatch for parents={parents}\nnew=\n{new}\nold=\n{old}"
    # New impl must be strictly block-lower-triangular (j < i).
    n = len(parents)
    if n > 0:
        assert not torch.any(torch.triu(new)), "ancestors must not include self or later indices"


def test_empty():
    m = build_ancestor_matrix_optimized([], torch.device("cpu"))
    assert m.shape == (0, 0)
    assert m.dtype == torch.bool


def test_depth3_width2_static_tree():
    # BloomBee's hardcoded default: depth=3, width=2 — parents are pre-order DFS.
    parents = [-1, 0, 0, 1, 1, 2, 2]
    _check(parents)


def test_linear_chain():
    parents = [-1, 0, 1, 2, 3, 4]
    _check(parents)


def test_wide_root_only():
    parents = [-1] + [0] * 10
    _check(parents)


def test_unbalanced_deep():
    # Mix of depths to exercise the walk termination.
    parents = [-1, 0, 1, 2, 0, 4, 5, 5, 7, 8]
    _check(parents)


def test_tree_attention_mask_has_self_attention():
    parents = [-1, 0, 0, 1, 2]
    mask = build_incremental_tree_attention_mask(
        past_len=0, tree_len=len(parents), parent_indices=parents, device=torch.device("cpu")
    )
    assert mask.shape == (1, 5, 5)
    # Diagonal must be True (each node attends to itself).
    diag = torch.diagonal(mask[0])
    assert diag.all()
