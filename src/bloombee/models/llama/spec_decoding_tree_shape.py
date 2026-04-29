"""Dynamic tree-shape policies for speculative decoding (EAGLE-2-style).

Implements the global-budget + confidence-threshold expansion rule from:

- Li, Y. et al. "EAGLE-2: Faster Inference of Language Models with
  Dynamic Draft Trees." https://arxiv.org/abs/2406.16858

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

Wired into ``MultiSSMDrafter._build_trees_batched`` via the optional
``tree_budget`` / ``tree_min_log_prob`` args. When neither is set the drafter
falls back to the original full-grid (depth × width) expansion, which keeps
pre-existing tests token-identical.
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


# ---------------------------------------------------------------------------
# Sequoia (arXiv 2402.12374) static-but-non-uniform tree shape
# ---------------------------------------------------------------------------
#
# Sequoia takes a per-depth acceptance histogram (from a calibration run)
# and chooses the per-depth branching factors that maximize expected
# accepted tokens per tree, under a total tree-size budget. Unlike EAGLE-2
# (online/budgeted expansion) and the default fixed-grid expansion, the
# Sequoia plan is a static ``List[int]`` of per-depth widths that the
# drafter follows mechanically — no per-step ranking cost at inference.


def sequoia_optimize_widths(
    per_depth_accept: Sequence[float],
    *,
    max_total_nodes: int,
    max_depth: int | None = None,
) -> List[int]:
    """Per-depth beam widths that maximize expected accepted tokens.

    ``widths[d]`` is the per-parent fanout at depth d. Total tree nodes =
    ``w0 + w0*w1 + w0*w1*w2 + ...``; this grows multiplicatively, so the
    cost of bumping widths[d] by 1 is ``prod(widths[:d])`` new leaves at
    that depth (plus children beneath once deeper depths are widened).

    We use greedy-by-marginal-expected-accept-gain per unit cost. The
    marginal value of the k-th sibling at depth d is
    ``reach(d) * accept_d * (1 - accept_d) ** (k-1)`` additional expected
    accepted tokens, and the cost is the number of new nodes added at
    depth d+1 alone — ``prod(widths[:d])``. This is a standard knapsack
    relaxation; for monotone-decreasing gains per depth-knob it's within
    ε of the Sequoia DP in practice and is way simpler to implement.
    """
    import heapq

    if max_depth is None:
        max_depth = len(per_depth_accept)
    max_depth = min(max_depth, len(per_depth_accept))
    if max_depth <= 0:
        return []
    widths = [0] * max_depth

    def _reach_prob(depth: int) -> float:
        # Probability that at least one path reaches depth ``depth``.
        p = 1.0
        for d in range(depth):
            if widths[d] == 0:
                return 0.0
            p *= 1.0 - (1.0 - per_depth_accept[d]) ** widths[d]
        return p

    def _bump_cost(depth: int) -> int:
        # Nodes added when widths[depth] is bumped by 1. That bump puts one
        # new node at every parent at depth ``depth`` (there are
        # ``prod(widths[:depth])`` such parents), and each of those new
        # nodes carries a subtree whose size is ``1 + w_{d+1} + w_{d+1}*w_{d+2} + ...``.
        parents_at_d = 1
        for d in range(depth):
            parents_at_d *= max(widths[d], 1)
        subtree = 1
        running = 1
        for d in range(depth + 1, max_depth):
            running *= max(widths[d], 0)
            if running == 0:
                break
            subtree += running
        return parents_at_d * subtree

    def _marginal_ratio(depth: int) -> float:
        a = per_depth_accept[depth]
        if a <= 0:
            return 0.0
        gain = a * (1.0 - a) ** widths[depth] * _reach_prob(depth)
        cost = max(_bump_cost(depth), 1)
        return gain / cost

    heap: list[tuple[float, int]] = []
    # Only depth 0 has nonzero reach initially.
    heapq.heappush(heap, (-_marginal_ratio(0), 0))

    remaining = max_total_nodes
    while remaining > 0 and heap:
        neg_r, d = heapq.heappop(heap)
        ratio = -neg_r
        if ratio <= 0.0:
            break
        # Gain may be stale if widths changed; recompute.
        fresh = _marginal_ratio(d)
        if fresh < ratio - 1e-12:
            if fresh > 0:
                heapq.heappush(heap, (-fresh, d))
            continue
        cost = _bump_cost(d)
        if cost > remaining:
            # This bump doesn't fit in the remaining budget — skip without
            # re-pushing; a future state change can re-seed via the
            # neighbor-push below when other depths widen.
            continue
        widths[d] += 1
        remaining -= cost
        for dd in (d, d + 1):
            if 0 <= dd < max_depth:
                new_r = _marginal_ratio(dd)
                if new_r > 0.0:
                    heapq.heappush(heap, (-new_r, dd))

    while widths and widths[-1] == 0:
        widths.pop()
    return widths


@dataclass
class AcceptanceHistogram:
    """Per-depth accept/seen counts accumulated across decode steps.

    ``seen[d]`` counts candidate draft edges at depth d that were checked
    against the target; ``accepted[d]`` counts how many passed. The ratio
    feeds Sequoia's DP.
    """
    seen: List[int]
    accepted: List[int]

    @classmethod
    def empty(cls, max_depth: int) -> "AcceptanceHistogram":
        return cls(seen=[0] * max_depth, accepted=[0] * max_depth)

    def record(self, depth: int, was_accepted: bool) -> None:
        if depth >= len(self.seen):
            pad = depth + 1 - len(self.seen)
            self.seen.extend([0] * pad)
            self.accepted.extend([0] * pad)
        self.seen[depth] += 1
        if was_accepted:
            self.accepted[depth] += 1

    def acceptance_rates(self, *, floor: float = 0.02) -> List[float]:
        out = []
        for s, a in zip(self.seen, self.accepted):
            out.append(max(floor, (a / s) if s > 0 else floor))
        return out

    def to_dict(self) -> dict:
        return {"seen": list(self.seen), "accepted": list(self.accepted)}

    @classmethod
    def from_dict(cls, d: dict) -> "AcceptanceHistogram":
        return cls(seen=list(d["seen"]), accepted=list(d["accepted"]))
