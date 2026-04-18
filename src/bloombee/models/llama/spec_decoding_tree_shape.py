"""Dynamic tree-shape policies for speculative decoding (EAGLE-2-style).

The default BloomBee drafter expands a fixed (depth, width) grid: at every
layer every frontier node gets ``beam_width`` children regardless of how
confident the path already is. For long-tail distributions this wastes
compute on low-probability branches that will almost certainly reject at
verify time.

EAGLE-2 addresses this with a global budget: across the whole tree, only
the top-B nodes (by cumulative path log-prob) are kept and expanded, and
low-confidence branches are pruned early. This module provides the pure
primitive so callers can swap the fixed expansion for a budgeted one
without touching the rest of the drafter orchestration.

Not yet wired into MultiSSMDrafter; lives here as a drop-in that tests
can exercise and future drafter code can import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class FrontierNode:
    """A node on the expansion frontier, with running path log-probability.

    ``node_handle`` is caller-opaque (callers typically pass a ``TreeNode``
    reference); this primitive only uses it for identity.
    """
    node_handle: object
    path_log_prob: float
    depth: int


def select_topk_by_logprob(
    frontier: Sequence[FrontierNode],
    budget: int,
) -> List[FrontierNode]:
    """Keep the top-``budget`` nodes by cumulative path log-probability.

    Ties are broken by insertion order (stable sort). Used to enforce a
    global cap on tree size regardless of depth.
    """
    if budget <= 0:
        return []
    if len(frontier) <= budget:
        return list(frontier)
    # Stable sort: Python's sort is stable, so equal log-probs preserve order.
    ranked = sorted(frontier, key=lambda n: -n.path_log_prob)
    return ranked[:budget]


def prune_below_threshold(
    frontier: Sequence[FrontierNode],
    min_log_prob: float,
) -> List[FrontierNode]:
    """Drop nodes whose running path log-prob is below ``min_log_prob``.

    Lets callers terminate low-confidence branches before investing another
    layer of drafter forward pass on them.
    """
    return [n for n in frontier if n.path_log_prob >= min_log_prob]


def budgeted_expand_plan(
    frontier: Sequence[FrontierNode],
    *,
    budget: int,
    min_log_prob: float | None = None,
) -> List[FrontierNode]:
    """Combine threshold pruning and top-K selection.

    Pipeline: optional threshold cull → top-K by log-prob. The returned list
    is the set of nodes the caller should expand at the next depth step.
    """
    pool = list(frontier)
    if min_log_prob is not None:
        pool = prune_below_threshold(pool, min_log_prob)
    return select_topk_by_logprob(pool, budget)


def child_logprob(parent_path_logprob: float, child_prob: float) -> float:
    """Running log-prob update for a candidate child.

    Guards against ``log(0)`` — a zero-prob child yields ``-inf``, which is
    correctly ordered by any downstream sort. Callers can filter these out.
    """
    import math

    if child_prob <= 0:
        return float("-inf")
    return parent_path_logprob + math.log(child_prob)
