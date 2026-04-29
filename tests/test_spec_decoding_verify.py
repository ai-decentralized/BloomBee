"""Unit tests for SpecInfer / Sequoia stochastic rejection sampling.

Verifies the pure-math primitive introduced for Phase 3. Argmax verify stays
in place; this kicks in only on the ``do_sample=True`` branch. Tests cover:

1. Deterministic edge cases: p==q (always accept), q=0 (forced accept),
   p=0 q>0 (must reject).
2. Residual formula: (p-q)+ properly normalized, falls back to p when
   everywhere zero.
3. Calibration: over many draws, accept rate converges to min(1, p/q).
4. Path walk: stops at first reject and commits resampled bonus token.
"""

import torch

from bloombee.models.llama.spec_decoding_verify import (
    EdgeVerifyResult,
    residual_distribution,
    verify_edge,
    verify_path,
)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum()


def test_residual_identity_when_disjoint():
    # Draft puts all mass on token 0, target on token 1 → residual == target.
    vocab = 4
    p = torch.tensor([0.0, 1.0, 0.0, 0.0])
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    res = residual_distribution(p, q)
    assert torch.allclose(res, p)


def test_residual_zero_fallback_to_target():
    # When draft dominates everywhere the residual is all zeros; impl must
    # fall back to p to avoid a degenerate multinomial.
    p = torch.tensor([0.3, 0.7])
    q = torch.tensor([0.6, 0.4])  # q[0] > p[0] and q[1] < p[1] so (p-q)+ = [0, 0.3]
    res = residual_distribution(p, q)
    # Non-degenerate case here — just verify it sums to 1 and is non-negative.
    assert torch.isclose(res.sum(), torch.tensor(1.0))
    assert (res >= 0).all()

    # Now an actually all-zero residual:
    p2 = torch.tensor([0.3, 0.7])
    q2 = torch.tensor([0.9, 0.9])  # q > p everywhere → residual is all zeros
    res2 = residual_distribution(p2, q2)
    # Fallback should be p normalized (p already sums to 1, so res2 == p).
    assert torch.allclose(res2, p2, atol=1e-6)


def test_edge_always_accept_when_p_equals_q():
    vocab = 4
    p = torch.tensor([0.25] * vocab)
    q = torch.tensor([0.25] * vocab)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        r = verify_edge(p, q, draft_token=2, generator=gen)
        assert r.accepted
        assert r.sampled_token == 2


def test_edge_forced_accept_when_draft_q_is_zero():
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([1.0, 0.0])  # draft puts 0 mass on token 1
    r = verify_edge(p, q, draft_token=1)
    # Per our impl, q=0 is treated as forced accept (no division by zero).
    assert r.accepted
    assert r.sampled_token == 1


def test_edge_must_reject_when_p_zero_and_q_positive():
    # p[draft]=0, q[draft]>0 → accept_prob = 0/q = 0 → always reject.
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.3, 0.7])
    gen = torch.Generator().manual_seed(42)
    for _ in range(30):
        r = verify_edge(p, q, draft_token=1, generator=gen)
        assert not r.accepted
        # Residual = (p-q)+ / norm = ([0.7, 0]+) / 0.7 = [1, 0] → always token 0.
        assert r.sampled_token == 0


def test_edge_acceptance_rate_converges_to_min_one_p_over_q():
    # Statistical calibration: over many trials acceptance rate ≈ min(1, p/q).
    vocab = 3
    p = torch.tensor([0.2, 0.5, 0.3])
    q = torch.tensor([0.5, 0.3, 0.2])
    draft_token = 0  # p/q = 0.2/0.5 = 0.4
    expected = 0.4

    gen = torch.Generator().manual_seed(123)
    n = 5000
    hits = 0
    for _ in range(n):
        r = verify_edge(p, q, draft_token=draft_token, generator=gen)
        if r.accepted:
            hits += 1
    rate = hits / n
    assert abs(rate - expected) < 0.03, f"accept rate {rate} not near {expected}"


def test_verify_path_stops_at_first_reject():
    # Craft target/draft so edge 0 always accepts (p==q), edge 1 always rejects
    # (p[draft]=0), edge 2 would succeed if we got there.
    vocab = 2
    draft_tokens = torch.tensor([0, 1, 0])

    p = torch.stack([
        torch.tensor([0.5, 0.5]),
        torch.tensor([1.0, 0.0]),  # residual will put all mass on token 0
        torch.tensor([0.5, 0.5]),
    ])
    q = torch.stack([
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 0.5]),
    ])

    committed, accepted = verify_path(p, q, draft_tokens)
    # Edge 0 accepted with ratio 0.5/0.5=1; edge 1 rejects (p[1]=0), bonus=0.
    assert accepted == 1
    # committed = [accepted_token, residual_resample]. length = accepted + 1.
    assert len(committed) == 2
    assert committed[0] == 0
    assert committed[1] == 0  # forced by residual


def test_verify_path_full_accept_emits_bonus_from_target():
    vocab = 2
    draft_tokens = torch.tensor([0, 1])
    p = torch.stack([
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ])
    q = torch.stack([
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ])

    committed, accepted = verify_path(p, q, draft_tokens)
    assert accepted == 2
    # committed = [t0, t1, bonus], bonus sampled from target_probs[-1]=[0,1] → 1.
    assert committed[:2] == [0, 1]
    assert committed[2] == 1
