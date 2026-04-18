"""Unit tests for EAGLE-2-style dynamic tree-shape primitives.

Covers top-K selection (stable tie-breaking), threshold pruning, combined
budgeted-expand pipeline, and log-prob accumulation edge cases.
"""

import math

from bloombee.models.llama.spec_decoding_tree_shape import (
    FrontierNode,
    budgeted_expand_plan,
    child_logprob,
    prune_below_threshold,
    select_topk_by_logprob,
)


def _mk(tag, lp, depth=1):
    return FrontierNode(node_handle=tag, path_log_prob=lp, depth=depth)


def test_topk_keeps_best_when_budget_smaller_than_frontier():
    frontier = [_mk("a", -1.0), _mk("b", -0.1), _mk("c", -2.0), _mk("d", -0.5)]
    picked = select_topk_by_logprob(frontier, budget=2)
    tags = [n.node_handle for n in picked]
    assert tags == ["b", "d"]


def test_topk_returns_all_when_budget_exceeds_frontier():
    frontier = [_mk("a", -1.0), _mk("b", -0.5)]
    picked = select_topk_by_logprob(frontier, budget=10)
    assert len(picked) == 2


def test_topk_zero_budget_returns_empty():
    frontier = [_mk("a", -0.1)]
    assert select_topk_by_logprob(frontier, budget=0) == []


def test_topk_stable_on_ties():
    # Equal log-probs must preserve insertion order (stable sort).
    frontier = [_mk("first", -0.5), _mk("second", -0.5), _mk("third", -0.5)]
    picked = select_topk_by_logprob(frontier, budget=2)
    assert [n.node_handle for n in picked] == ["first", "second"]


def test_prune_below_threshold():
    frontier = [_mk("a", -3.0), _mk("b", -0.5), _mk("c", -1.5)]
    kept = prune_below_threshold(frontier, min_log_prob=-2.0)
    tags = [n.node_handle for n in kept]
    assert tags == ["b", "c"]


def test_budgeted_expand_combines_prune_and_topk():
    frontier = [
        _mk("hi",   -0.2),
        _mk("mid",  -1.0),
        _mk("low",  -5.0),  # will get pruned by threshold
        _mk("hi2",  -0.1),
    ]
    plan = budgeted_expand_plan(frontier, budget=2, min_log_prob=-2.0)
    tags = [n.node_handle for n in plan]
    assert tags == ["hi2", "hi"]


def test_budgeted_expand_no_threshold_is_pure_topk():
    frontier = [_mk("a", -0.3), _mk("b", -0.1), _mk("c", -0.8)]
    plan = budgeted_expand_plan(frontier, budget=2)
    assert [n.node_handle for n in plan] == ["b", "a"]


def test_child_logprob_accumulates_additively():
    assert math.isclose(child_logprob(0.0, 0.5), math.log(0.5))
    assert math.isclose(child_logprob(math.log(0.5), 0.5), math.log(0.25))


def test_child_logprob_zero_yields_neg_infinity():
    assert child_logprob(-0.1, 0.0) == float("-inf")
    # And such a child sorts to the bottom.
    frontier = [_mk("alive", -0.5), _mk("dead", float("-inf"))]
    picked = select_topk_by_logprob(frontier, budget=1)
    assert picked[0].node_handle == "alive"
