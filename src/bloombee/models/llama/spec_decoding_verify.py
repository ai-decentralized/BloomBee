"""Stochastic rejection sampling for tree speculative decoding.

Implements the per-edge rejection sampling rule from:

- Miao, X. et al. "SpecInfer: Accelerating Large Language Model Serving
  with Tree-based Speculative Inference and Verification."
  https://arxiv.org/abs/2305.09781
- Chen, Z. et al. "Sequoia: Scalable, Robust, and Hardware-aware
  Speculative Decoding." https://arxiv.org/abs/2402.12374

Rule: accept draft token ``t`` with probability
``min(1, p_target(t) / p_draft(t))``. On reject, resample from the residual
distribution ``(p_target - p_draft)+`` (normalized). When every edge on a
path is accepted, emit one bonus token sampled from the final target
distribution.

This module stays pure — no batching glue, no tree traversal — so it can be
unit-tested against known distributions independent of the rest of the
speculative generate loop. The caller is responsible for walking paths
and stitching the results back into KV cache position ids.

The existing argmax path (``speculative_model._extract_best_verified_paths_fixed``)
remains the default when ``do_sample=False`` and is token-identical to greedy
decoding on the target model. This primitive activates only on the stochastic
branch (``do_sample=True``) and is provably distribution-equivalent to
sampling directly from the target model (see SpecInfer §3.2 / Sequoia §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class EdgeVerifyResult:
    """Outcome of verifying a single draft token at one tree edge."""
    accepted: bool
    sampled_token: int


def residual_distribution(p_target: torch.Tensor, p_draft: torch.Tensor) -> torch.Tensor:
    """Compute the residual ``(p_target - p_draft)+`` normalized to sum to 1.

    If the residual is zero everywhere (p_draft dominates p_target everywhere),
    fall back to p_target — this matches the SpecInfer reference impl and
    avoids a degenerate all-zero distribution.
    """
    residual = torch.clamp(p_target - p_draft, min=0.0)
    total = residual.sum()
    if total <= 0:
        return p_target / p_target.sum().clamp(min=1e-20)
    return residual / total


def verify_edge(
    p_target: torch.Tensor,
    p_draft: torch.Tensor,
    draft_token: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> EdgeVerifyResult:
    """Run SpecInfer rejection sampling on a single edge.

    Args:
        p_target: target-model probability vector, shape (vocab,). Must be
            non-negative and sum to 1 (or be normalized by the caller).
        p_draft: draft-model probability vector, shape (vocab,).
        draft_token: the token id proposed by the drafter at this edge.
        generator: optional torch.Generator for deterministic tests.

    Returns:
        EdgeVerifyResult(accepted=bool, sampled_token=int). When accepted,
        ``sampled_token == draft_token``; when rejected, ``sampled_token`` is
        drawn from the residual distribution.
    """
    assert p_target.ndim == 1 and p_draft.ndim == 1, "distributions must be 1D"
    assert p_target.shape == p_draft.shape, "target/draft shapes must match"
    assert 0 <= draft_token < p_target.shape[0], "draft_token out of vocab"

    q = p_draft[draft_token]
    p = p_target[draft_token]

    # SpecInfer acceptance ratio. If q == 0, the drafter "should not have"
    # proposed this token; treat as forced accept to match reference impl.
    if q <= 0:
        return EdgeVerifyResult(accepted=True, sampled_token=int(draft_token))

    accept_prob = torch.clamp(p / q, max=1.0)
    u = torch.rand(1, device=p_target.device, generator=generator).item()
    if u < accept_prob.item():
        return EdgeVerifyResult(accepted=True, sampled_token=int(draft_token))

    # Rejected: sample from residual.
    res = residual_distribution(p_target, p_draft)
    sampled = torch.multinomial(res, num_samples=1, generator=generator).item()
    return EdgeVerifyResult(accepted=False, sampled_token=int(sampled))


def verify_path(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[list[int], int]:
    """Walk one tree path applying rejection sampling until the first reject.

    SpecInfer / Sequoia semantics: stop at the first rejected edge, emit the
    resampled token as the final committed token, return the accepted prefix
    plus that bonus token. The caller stitches the returned ``(tokens, len)``
    into KV cache position ids.

    Args:
        target_probs: (path_len, vocab). Row i is the target distribution at
            edge i, normalized.
        draft_probs: (path_len, vocab). Draft distribution at each edge.
        draft_tokens: (path_len,). Draft token id at each edge.

    Returns:
        (committed_tokens, accepted_len) where ``accepted_len`` is the number
        of draft tokens accepted (i.e., the position of the first reject, or
        ``path_len`` if every edge was accepted). The emitted bonus token at
        the end comes from either the resampled residual or, on full accept,
        the final target distribution.
    """
    assert target_probs.ndim == 2 and draft_probs.ndim == 2
    path_len = target_probs.shape[0]
    assert draft_tokens.shape == (path_len,)

    committed: list[int] = []
    accepted_len = 0
    for i in range(path_len):
        result = verify_edge(
            target_probs[i],
            draft_probs[i],
            int(draft_tokens[i].item()),
            generator=generator,
        )
        if result.accepted:
            committed.append(result.sampled_token)
            accepted_len += 1
        else:
            committed.append(result.sampled_token)
            return committed, accepted_len

    # Full path accepted: emit an extra token sampled from the last target dist.
    # Caller can choose to use target_probs[-1] directly, but for consistency
    # with SpecInfer we emit one bonus token from target.
    bonus = torch.multinomial(target_probs[-1], num_samples=1, generator=generator).item()
    committed.append(int(bonus))
    return committed, accepted_len
