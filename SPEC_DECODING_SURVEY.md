# Speculative Decoding Frontier Survey (as of 2026-04)

Current BloomBee SD: static depth-3 width-2 tree, argmax sequential verify, dense O(n²) ancestor mask, coarse KV rollback, hardcoded `session_max_length=624`. Roughly vLLM 2.x-era. vLLM 4.x line brought in EAGLE-2/3, dynamic tree shaping, tree-attention verification with proper rejection sampling.

## Algorithms one-liner matrix

| # | Algo | arXiv | Novel idea | vs. what | Needs tail features? |
|---|------|-------|------------|----------|----------------------|
| 1 | Medusa / Medusa-2 | [2401.10774](https://arxiv.org/abs/2401.10774) | K parallel decoding heads on frozen backbone | autoregressive single-token | **YES** — heads sit on backbone's final hidden state |
| 2 | Hydra | [2402.05109](https://arxiv.org/abs/2402.05109) | Sequentially-dependent draft heads (condition on prior draft tokens) | Medusa's independent heads | **YES** — same as Medusa |
| 3 | EAGLE-1 | [2401.15077](https://arxiv.org/abs/2401.15077) | Draft at second-to-top feature level | token-level drafting | **YES** — needs 2nd-to-last layer feature |
| 4 | EAGLE-2 | [2406.16858](https://arxiv.org/abs/2406.16858) | **Dynamic tree** via draft-model confidence → acceptance | EAGLE-1 static tree | YES |
| 5 | EAGLE-3 | [2503.01840](https://arxiv.org/abs/2503.01840) | Direct token prediction + multi-layer feature fusion (TTT) | EAGLE-2's single-top-layer bottleneck | YES |
| 6 | Sequoia | [2402.12374](https://arxiv.org/abs/2402.12374) | Hardware-aware DP over tree shape; probabilistic accept | static tree sizes | Drafter is external → **NO** tail-feature dep. |
| 7 | SpecInfer | [2305.09781](https://arxiv.org/abs/2305.09781) | Token-tree verifier; external SSM drafter | single-candidate spec dec | **NO** — external drafter, logits only |
| 8 | Lookahead | [2402.02057](https://arxiv.org/abs/2402.02057) | Jacobi iteration + N-gram pool; no drafter needed | draft-target pair | **NO** — self-speculative on target's own forward |
| 9 | Self-Spec (Zhang) | [2309.08168](https://arxiv.org/abs/2309.08168) | Skip target's own middle layers for drafting | external drafter | Uses subset of target layers → **NO auxiliary state**, but needs layer skipping mid-pipe |
| 10 | Token Recycling | [2408.08696](https://arxiv.org/abs/2408.08696) | Store rejected tokens in adjacency matrix, BFS-build tree | any trained drafter | **NO** — purely runtime, no heads, no features |

## Dynamic tree (EAGLE-2 / Sequoia line)

- **Static** (EAGLE-1, Medusa, BloomBee today): pre-commit to width × depth. Wastes computation on branches that will be rejected; under-explores easy tokens.
- **Dynamic EAGLE-2**: at each draft step, the drafter's own logit confidence is calibrated to acceptance probability. Build tree breadth-first; expand only nodes whose confidence × ancestor-acceptance exceeds a threshold. Prunes doomed branches before they cost compute.
- **Dynamic Sequoia**: pre-compute an optimal tree topology per (GPU, model, sequence length) via DP over `f(tree_size, depth) → expected_acceptance_length / step_time`. Tree shape adapts to the hardware, not runtime signal. Complementary to EAGLE-2's runtime pruning.

## Tree attention + batched verification

Key paper: **SpecInfer (2305.09781)** and EAGLE-2/3 appendix. O(tree_size × seq_len) mask built once per step; each node attends to its ancestor path via a block-lower-triangular mask. Verification: run target on all tree tokens in one forward; use probabilistic rejection sampling (resample from `(target_logit - α·draft_logit)⁺`) per ancestor-to-child edge; take the longest accepted path.

## Distributed-pipeline fit (Petals-style, tail on remote peer)

| Algo | Fit for BloomBee | Why |
|------|------------------|-----|
| Medusa / Hydra / EAGLE-{1,2,3} | **Bad** | Drafter consumes target's near-final hidden state. Would force every spec step to roundtrip through the tail peer. Also forces training an aux head tied to this model. |
| SpecInfer, Sequoia | **Good** | External SSM drafter runs anywhere (ideally on first peer). Tree of token IDs ships through pipeline like any other prefill. |
| Lookahead | **Good** | No drafter, no extra state. N-gram pool is per-session on the client. Jacobi iteration uses target's own forward. Minimal cross-peer coordination. |
| Self-Speculative (layer skip) | **Good, natural** | Skipping middle layers → on a pipeline with K peers, just skip some layers on each peer. No new state. |
| Token Recycling | **Excellent (ops-wise)** | No heads, no training, 2MB adjacency matrix per session. But speedup ceiling is lower (~2×). |

## Empirical numbers on LLaMA-7B

- EAGLE-2: 3.05–4.26× vs. HF baseline (A100 numbers, no V100-specific figures). V100 without flash-attn v2 likely loses 20–30% of this.
- Sequoia: up to 4.04× on LLaMA-7B A100. No V100 figures.
- Token Recycling: ~2× across sizes, training-free.
- Lookahead: up to 1.8× on MT-bench.
- Medusa-2: 2.3–3.6×.

None of these benchmark specifically on 16GB V100. Expect the single-GPU wins to be mediocre; the real win for BloomBee is at the **distributed-pipeline level**, where speculation amortizes cross-peer latency.

## Recommendation: ranked shortlist

### Pick 1 — **Sequoia + SpecInfer-style verification** (primary)
- arXiv: [2402.12374](https://arxiv.org/abs/2402.12374), [2305.09781](https://arxiv.org/abs/2305.09781)
- Why it fits BloomBee: external drafter (we keep a cheap SSM or even a 68M-param model hosted on the first peer), ships token IDs through pipeline like normal traffic, probabilistic rejection sampling replaces our argmax-verify bug, DP-optimized tree shape per GPU beats our hardcoded `depth=3 width=2`. No tail-peer hidden state coupling.
- Primitives we need: (a) external drafter hookup, (b) DP-optimal tree shape selector (offline one-time per model+GPU), (c) block-lower-triangular tree-attention mask builder, (d) probabilistic rejection sampler `P(accept) = min(1, p/q)`, (e) KV rollback by page (aligns with Phase 2 paged KV).

### Pick 2 — **Lookahead** (backup / ensemble)
- arXiv: [2402.02057](https://arxiv.org/abs/2402.02057)
- Why it fits: no drafter, no training, trivial to deploy in a distributed setting. Can coexist with Sequoia as a fallback when the drafter is cold-starting or unavailable. Strong for code/repetitive generation.
- Primitives: (a) Jacobi iteration window, (b) N-gram pool per-session, (c) verification via target's own forward — reuses Sequoia's tree-attention mask builder.

### Pick 3 — **EAGLE-2 (deferred)**
- arXiv: [2406.16858](https://arxiv.org/abs/2406.16858)
- Why deferred: best single-GPU numbers but needs near-final-layer features, which in BloomBee means cross-peer roundtrip per draft step. Revisit after Phase 2 ships and if we add a "speculation is centralized at tail peer" mode.

## What this means for the BloomBee plan

- Drop the "Lookahead first, EAGLE deferred" framing from the original plan. **Sequoia-style external drafter + tree-attention verify** is a better primary because it ships through the pipeline like normal tokens and cleanly decouples from layer-slice ownership.
- All three picks need **per-page KV rollback**, which only becomes clean after Phase 2. Phase 2 → Phase 3 ordering from Codex holds.
- Do NOT delete `speculative_model.py`, `spec_decoding_drafter.py`, `spe_dec_tree.py`. Keep their scaffolding (session management, draft-target plumbing, RPC metadata) — surgically upgrade the algorithm inside:
  - `spe_dec_tree.py:170-185` → replace dense O(n²) ancestor matrix with packed CSR-style parent pointers + block-lower-triangular mask.
  - `speculative_model.py:619-631` → replace argmax sequential verify with vectorized rejection sampling across the tree.
  - `speculative_model.py:364-388` → replace coarse trim with per-page rollback (depends on Phase 2).
  - `spec_decoding_drafter.py` → add dynamic tree shape selection (EAGLE-2 confidence-based) OR load a pre-computed Sequoia DP tree shape for the active (GPU, model).
  - Drop `session_max_length=624` hardcode.
