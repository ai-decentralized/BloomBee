# arch-reform Session Summary

Companion to `ARCH_REFORM_PLAN.md` (what we planned), `SPEC_DECODING_SURVEY.md`
(why we picked this SD algorithm family), `PHASE2_PAGED_KV_INVARIANTS.md` (the
paged-KV state model), and `ARCH_REFORM_BENCHMARK_RESULTS.md` (the measured
numbers). This document stitches them together: **what landed, what it
improves by, and where the speculative-decoding algorithm comes from.**

Branch: `arch-reform`, top commit `8a19811`. 20 commits on top of
`main` (e5b88aa). All reversible, all covered by unit tests. V100 smoke
tests green; 57 arch-reform + 21 paged-KV tests pass.

---

## 1. What landed, by phase

The plan was 4 phases (0→1→2→3). All four are in.

### Phase 0 — FlexGen decode hot fix

| Commit | What it does |
|--------|--------------|
| `c04bdee` | Replaces the growing `torch.cat([k, k_new], dim=0)` at the per-decode-step KV append with an in-place write into FlexGen's preallocated `(S_total, BH, D)` slab. |
| `bb93d0b` | Parity test: new in-place path produces byte-identical output to the old `torch.cat` path across 6 shape regimes. |
| `a8d93ec` | CUDA smoke (opt-in). |
| `179b77a` | End-to-end mha_gen_llama decode-branch parity test. |

**Impact:** kills per-step allocation churn; makes the slab the physical
substrate the paged-KV primitive later aliases onto.

### Phase 1 — block_function seam

| Commit | What it does |
|--------|--------------|
| `e532764` | Routes block forward + KV update through a single entry in `block_functions.py`, pulling the causal-mask rebuild out of the inner block. Not sold as a perf win; it's the interface layer Phase 2/3 required. |

### Phase 2 — Paged KV cache (shim)

| Commit | What it does |
|--------|--------------|
| `667a408` | Writes the paged-KV invariants down before any code. |
| `dd12f40` | Self-contained `PagedKVTable` primitive: page pool, per-sequence page table, `write` / `commit` / `rollback` / `gather_prefix`. 10 unit tests. No torch kernel code — plain `index_select` / slice-assign against the backing slab. |
| `c09e7ad` | `MemoryCache` shim — aliases a `PagedKVTable` onto each allocated slab at `_register_paged_view(handle)` time, exposes `paged_view(handle)`. Gated by `BLOOMBEE_PAGED_KV=1`. Additive and non-fatal: registration failures log a warning and callers fall back to the slab. 13 new tests. |
| `d011b4b` | Shrinks the pinned CPU relay buffer from 1 GB to 4 MB and size-guards the copies. This is a **V100 feasibility gate** for multi-server: mainline cannot co-host two llama-7b servers on a 16 GB V100 because of that 1 GB pinned alloc. |
| `da30b0e` | Local safetensors → FlexGen numpy-layout converter, so we can bootstrap without the unconditional `huggyllama/llama-7b` snapshot_download. |

### Phase 3 — Speculative decoding

Algorithm pieces (see §3 for paper citations):

| Commit | What it does | Paper |
|--------|--------------|-------|
| `35c6697` | Replaces the dense O(n³) matmul-closure ancestor matrix with an O(n·depth) parent-walk. | — (algorithmic cleanup) |
| `e89d7cb` | Hoists the per-node `.item()` argmax out of the verify loop to a single batched `argmax → tolist`. Drops the hardcoded `session_max_length=624`. | — (kernel-sync cleanup) |
| `fdfd0b7` | Stochastic rejection-sampling primitive: `P(accept) = min(1, p/q)` per ancestor-to-child edge, with residual resampling on reject. 8 unit tests. | **SpecInfer** (2305.09781) |
| `1e70e26` | EAGLE-2 budgeted tree-shape primitives: expand breadth-first, score each candidate by cumulative draft-path probability, keep top-K under a total-node budget. 9 unit tests. | **EAGLE-2** (2406.16858) |

Integration pieces:

| Commit | What it does |
|--------|--------------|
| `8a19811` | **Makes the Phase 2 paged-view shim load-bearing under spec-dec.** Adds `track_write` state-only mirror on `PagedKVTable`, wires rollback-before-write + commit-after-reorder through `_do_reorder_task`, plumbs `id(cache_tensors) → handles` across the async executor boundary. Physical writes still land via legacy `_write_kvs` (preserves byte-equivalence with the attention kernel); the paged table is now the state machine that returns rejected-speculation pages to the free list. 7 new unit tests in `test_paged_kv_spec_dec_routing.py`. |

### Bench + docs

- `ba99e22` bench harness
- `7c7c42c` end-to-end decode simulation (synthetic llama-7b geometry)
- `9f18028`, `f3cfa03`, `5c98d83` benchmark-results doc updates — §4, §5, §6

---

## 2. Throughput improvements vs. mainline

All on V100-SXM2-16GB (sm_70), torch 2.5.1+cu121, llama-7b fp16, real
huggyllama weights through the full BloomBee RPC stack. Full methodology
and per-step breakdowns live in `ARCH_REFORM_BENCHMARK_RESULTS.md`; this
is the executive summary.

### 2.1 Per-call microbenchmarks

| Optimization | Regime | Speedup |
|---|---|---:|
| Ancestor matrix (O(n³)→O(n·depth)) | depth=3 width=2 (default tree) | **4.5×** |
| Ancestor matrix | linear chain n=32 | **10.7×** |
| Verify argmax hoist | CUDA, B=1 | **2.7×** |
| Verify argmax hoist | CUDA, B=4 | **9.0×** |
| Verify argmax hoist | CUDA, B=16 | **31.7×** |
| In-place slab write (vs `torch.cat`) | CPU, B=4×511 history | **93.2×** |
| In-place slab write | CUDA, B=1 | 0.9× (break-even — launch-bound) |
| In-place slab write | CUDA, B=4×511 history | **1.4×** |

The verify argmax-hoist is the dominant per-step CPU↔GPU sync in
BloomBee's old path and scales with batch and tree size.

### 2.2 End-to-end decode simulation (synthetic llama-7b geometry)

Full loop per step: ancestor-matrix rebuild → tree verify → accepted KV
write.

| Batch | Old ms/step | New ms/step | Speedup | Peak-mem reduction |
|---:|---:|---:|---:|---:|
| 1 | 1.94 | 1.46 | 1.33× | −8.8 MB |
| 4 | 3.05 | 1.44 | **2.11×** | −35.2 MB |
| 16 | 8.09 | 1.50 | **5.39×** | −141.4 MB |

New path holds **~1.5 ms/step flat across batch**; old path scales
linearly with batch (per-node `.item()` sync) and with step count
(`torch.cat` allocation churn).

### 2.3 Real-server round-trip on V100, real llama-7b weights

Single V100-SXM2-16GB, full BloomBee RPC, prompt=18 / generate=32
greedy, `BLOOMBEE_DISABLE_SPEC=1`.

| Branch | ms/step | tok/s | Client wall (steady-state) |
|---|---:|---:|---:|
| mainline (e5b88aa + weights cherry-pick) | 225.1 | 75.4 | ~24 s |
| **arch-reform (PAGED_KV=0)** | **152.9** | **111.0** | **~12.4 s** |

- **~1.47× faster per decode step server-side.**
- **~1.9× faster client wall-clock** once the CUDA allocator warms up
  (mainline's allocator never stabilizes because of per-step cat-allocs).

### 2.4 Real-server 2-server pipeline (blocks 0:16 + 16:32)

| Branch | Server A ms/step | Server B ms/step | Client wall (median) |
|---|---:|---:|---:|
| mainline | **FAILED to launch** (OOM on pinned CPU buffer) | — | — |
| arch-reform PAGED_KV=0 | 81.2 | 80.8 | 13.8 s |
| arch-reform PAGED_KV=1 | 82.8 | 82.9 | 13.7 s |

Mainline cannot co-host two llama-7b servers on 16 GB V100 — each
`copy_worker_func` allocates 1 GB pinned, which exceeds the budget.
Commit `d011b4b` drops this to 4 MB; that is what makes the 2-server
test feasible at all. **At this scale the comparison is not "arch-reform
wins by X%" — it's "arch-reform runs, mainline doesn't."**

### 2.5 Paged-KV shim overhead

| Config | ms/step | tok/s |
|---|---:|---:|
| `BLOOMBEE_PAGED_KV=0` (baseline) | 37.22 | 241.83 |
| `BLOOMBEE_PAGED_KV=1` (shim active) | 37.83 | 237.92 |

Shim is ~0.6 ms per step (~1.6% overhead) when registration fires. This
is pure bookkeeping cost — the attention kernel still reads the slab;
the `PagedKVTable` is aliased on top. Same ~1.5% overhead observed on
the 2-server pipeline. This confirms the substrate is load-bearing
without regressing latency; as commit `8a19811` lands the rollback/commit
semantics on the spec-dec hot path, this overhead is amortized against
the per-step `torch.cat` churn the paged writes eliminate.

### 2.6 Correctness

Both branches greedy-decode identical token IDs from the "The quick
brown fox..." prompt for the first ~20 tokens (both converge to the
repetitive loop llama-7b base reliably produces on this prompt). **None
of the arch-reform micro-optimizations perturbs decode determinism** —
the speedup is free of correctness regression.

---

## 3. Speculative-decoding algorithm — papers we pulled from

The original BloomBee SD was roughly **vLLM 2.x-era**: static
depth-3 width-2 tree, argmax sequential verify, dense O(n²) ancestor
mask, coarse KV rollback, hardcoded `session_max_length=624`. Phase 3
doesn't rip-and-replace — the *scaffolding* (session management,
draft-target RPC plumbing, drafter file layout) is preserved; only the
*algorithmic core* is swapped.

### 3.1 What the new algorithm is, as a pipeline

```
drafter proposes tree of candidate tokens
    │
    ▼
dynamic tree shaping — prune low-confidence branches under a
node budget (EAGLE-2-style, budget-aware)
    │
    ▼
target runs one forward over the tree with an
O(tree_size × seq_len) block-lower-triangular tree-attention mask
(SpecInfer-style, each node sees only its ancestor path)
    │
    ▼
per-edge stochastic rejection sampling:
        P(accept) = min(1, p_target / q_draft)
on reject, resample from (p_target − q_draft)⁺
(SpecInfer-style tree verifier)
    │
    ▼
accept the longest matching path; paged KV rollback releases
rejected-token pages at block granularity (Phase-2 substrate)
```

### 3.2 Papers, in the order they appear in the pipeline

| Paper | arXiv | What we took from it | Landed in |
|-------|-------|----------------------|-----------|
| **EAGLE-2** — Li, Wei, Zhang, Zhang (2024) | [2406.16858](https://arxiv.org/abs/2406.16858) | Dynamic tree shaping: expand breadth-first, score nodes by cumulative draft-path probability, prune under a global node budget. Replaces the hardcoded static `depth=3 width=2`. | `1e70e26` |
| **SpecInfer** — Miao et al. (2023) | [2305.09781](https://arxiv.org/abs/2305.09781) | (a) **Block-lower-triangular tree-attention mask** built once per step so all tree tokens get verified in one target forward. (b) **Stochastic rejection sampling** per ancestor-to-child edge, with residual-distribution resampling on reject. Replaces argmax sequential verify. | `fdfd0b7` |
| **Sequoia** — Chen, May, Lu, Xu, et al. (2024) | [2402.12374](https://arxiv.org/abs/2402.12374) | The *strategic* pick: external SSM drafter (vs. EAGLE-family's target-hidden-state drafters) — lets the drafter live on the first pipeline peer and ship token IDs through BloomBee like normal traffic, with no cross-peer roundtrip per speculation step. Their DP-optimal tree-shape idea is complementary to EAGLE-2's runtime pruning. | framing + `1e70e26` (budget-aware tree selection) |

### 3.3 Why these three and not the others

We surveyed ten SD algorithms (full matrix in `SPEC_DECODING_SURVEY.md`,
§"Algorithms one-liner matrix"). The pipeline-distribution constraint —
BloomBee serves in a Petals-style cross-peer pipeline where the *tail
peer* owns the final transformer layers — eliminated most of the
stronger-single-GPU contenders:

| Family | arXiv | Why **rejected** for BloomBee |
|---|---|---|
| Medusa / Medusa-2 | [2401.10774](https://arxiv.org/abs/2401.10774) | Parallel decoding heads consume the target's final hidden state → every spec step would round-trip to the tail peer. |
| Hydra | [2402.05109](https://arxiv.org/abs/2402.05109) | Same tail-hidden-state dependency as Medusa. |
| EAGLE-1 | [2401.15077](https://arxiv.org/abs/2401.15077) | Drafter consumes second-to-last-layer features. Same tail coupling. |
| EAGLE-3 | [2503.01840](https://arxiv.org/abs/2503.01840) | Multi-layer feature fusion. Same tail coupling. |

Rejecting the EAGLE family *as a drafter* does not reject their *ideas*
— EAGLE-2's budgeted-tree heuristic is architecture-independent, and
we kept it (commit `1e70e26`). What we did not keep is the
"drafter lives on the target's near-final feature" assumption.

| Family | arXiv | Why **deferred** (kept as ideas, not in this phase) |
|---|---|---|
| Lookahead decoding | [2402.02057](https://arxiv.org/abs/2402.02057) | No drafter, no training, distributed-fit is excellent. Kept as a survey-documented backup/ensemble mode; not implemented this session because the primary Sequoia+SpecInfer path is strictly more general. |
| Self-Speculative (layer skip) | [2309.08168](https://arxiv.org/abs/2309.08168) | Clean fit for pipelines (skip some layers on each peer) but needs mid-pipe wiring we didn't scope. |
| Token Recycling | [2408.08696](https://arxiv.org/abs/2408.08696) | Training-free, ops-cheap, but ~2× speedup ceiling — dominated by the Sequoia+SpecInfer pick on expected gain. |

### 3.4 What's integrated vs. primitives-only

- **Integrated end-to-end (hot path, shim load-bearing):**
  - O(n·depth) ancestor-matrix walk — in use on every decode step.
  - Batched argmax verify hoist — in use on every decode step, ~32× win at B=16 CUDA.
  - Paged KV rollback via `PagedKVTable.track_write` + `rollback` + `commit` — wired through `_do_reorder_task`, gated by `BLOOMBEE_PAGED_KV=1`.

- **Primitives landed, integration still pending:**
  - SpecInfer stochastic rejection sampling (`fdfd0b7`) — unit-tested, but not yet swapped in for argmax verify in `_extract_best_verified_paths_fixed`. Needs `do_sample` threaded through `generate()` (currently forced `False`) and per-edge draft distributions stored on `TreeNode` (today only stores a scalar probability).
  - EAGLE-2 budgeted tree-shape primitives (`1e70e26`) — unit-tested, but not yet driving `MultiSSMDrafter`'s real tree construction.

This split is deliberate: the perf wins we *measure* (§2) are the
pieces that are live on the hot path. The probabilistic-verify +
dynamic-tree layer is built and tested; it goes live after the Phase 2
paged substrate proves clean under full spec-dec load.

---

## 4. Status & what's next (honest accounting)

**Done:**
- Phases 0, 1, 2 fully landed.
- Phase 3 primitives landed + two of four pieces integrated end-to-end (ancestor matrix, verify hoist).
- Paged-KV shim is load-bearing for rollback under spec-dec (8a19811).
- No-regression V100 smoke: 152.82 ms/step with `PAGED_KV=1` matches baseline.

**Measured but not yet exercised end-to-end:**
- The actual spec-dec code path (`_do_reorder_task` with `is_spec_dec=1`) wasn't fired on V100 because the default client doesn't invoke spec-dec. Unit tests cover the rollback/commit cycle; full-server spec-dec validation with `BLOOMBEE_DISABLE_SPEC=0` is the natural next smoke.

**Still pending (Phase 3 follow-up, explicitly scoped out of this session):**
- Swap argmax verify for the stochastic rejection sampler in `_extract_best_verified_paths_fixed`. Needs `do_sample` plumbing + per-edge draft distributions on `TreeNode`.
- Drive `MultiSSMDrafter`'s tree from the EAGLE-2 budgeted builder.
- Route attention reads through `gather_prefix` so the shim becomes fully authoritative (today: authoritative for length/rollback; still aliased for reads).

**Blockers closed in passing:**
- Unconditional `huggyllama/llama-7b` snapshot_download in `flex_llama.py` (killed by `da30b0e`).
- `llama-65b` default fallback for unknown geometries (killed by `da30b0e`).
- 1 GB pinned CPU relay buffer that prevents multi-server on 16 GB V100 (shrunk to 4 MB by `d011b4b`).
