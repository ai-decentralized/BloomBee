# BloomBee Architecture Reform Plan

Based on code exploration + review with Codex GPT-5.x. Target validation: 16 GB V100 (sm_70, no bf16, no flash-attn v2) running LLaMA-7B.

## Pain points (verified in code)

| # | Pain | Evidence |
|---|------|----------|
| 1 | FlexGen attention: full `torch.cat` of growing KV per token | `src/bloombee/flexgen_utils/pytorch_backend.py:784-785` with preallocated slab already at `:471-476`. `load_cache`/`store_cache` no-op at `src/bloombee/models/llama/flex_llama.py:522-526`. |
| 2 | Old speculative decoding: argmax sequential verify, dense O(n²) tree mask, small γ, coarse rollback | `src/bloombee/models/llama/speculative_model.py:619-631`, `spe_dec_tree.py:170-185`, `speculative_model.py:364-388`, hardcoded `session_max_length=624` at :58. |
| 3 | Hot path split: select_cache → forward → update_cache crosses Python boundary 3× per step; causal mask rebuilt inside block per step | `src/bloombee/server/backend.py:341-348, 394-401, 444-449`; `src/bloombee/models/llama/block.py:850-874`. |
| 4 | No continuous batching: pool pops one task at a time; contiguous per-request KV; position_ids bound to request lifetime | `src/bloombee/server/task_pool.py:36, 124-137`; `memory_cache.py:257`; `backend.py:324-378`; mask `(B,L,L)` at `backend.py:363-364`. |

## Phase order (revised after Codex review)

**0 → 1 → 2 (paged KV + continuous batching) → 3 (speculative decoding)**

Original order had spec decoding before continuous batching. Codex pushed back: if spec decoding lands on today's contiguous-per-request KV and coarse rollback, it will be rewritten when paged KV ships. Defer spec until rollback + page ownership are explicit.

### Phase 0 — FlexGen decode hot fix (quick, low risk)

- Replace `torch.cat([k, k_new], dim=0)` at `pytorch_backend.py:784-785` with in-place write at `(src_s-1):src_s` into the preallocated slab. Return a `TorchTensor` wrapper of the relevant slice for attention.
- Skip ring buffer unless we also implement sliding-window eviction (different contract — don't couple).
- `load_cache`/`store_cache` hooks: activate only if offload/CPU-cache path is actually live. If dead, don't invest.
- Do NOT integrate `utils/cache_compat.py` here — FlexGen owns this cache layout. DynamicCache is the transformers-side story.

### Phase 1 — `block_function` single-pass API (interface, not perf)

- New entry `block_function(hidden_states, cache_handle, step_metadata) -> (output, new_cache_handle)` in `src/bloombee/server/block_functions.py`.
- Internal: select → attention + FFN → update, in one Python call.
- Move causal mask from `block.py:850-874` up to backend, build once per step.
- Designed around **logical cache handles** (sequence ids + per-seq length), not raw tensors — Phase 2 compatible without rewrite.
- Don't sell this phase as a speedup. Honest framing: this is the abstraction layer Phases 2 and 3 need.

### Phase 2 — Paged KV cache + continuous batching

Write invariants first:
- logical sequence length vs. physical page mapping
- accepted prefix length
- rollback semantics
- page ownership, free semantics
- distributed protocol: does a remote peer assume request X has a linear KV region or a page table?  
  (If linear, paging is NOT a local refactor — it's a protocol change. Audit this before writing code.)

Implementation:
1. Replace `memory_cache.py:257` contiguous alloc with page table (block_size=16).
2. Per-token `position_ids` from logical seq length (remove fixed `cache_len+offset` in `backend.py:324-378`).
3. Mask: swap `(B,L,L)` for per-token block-sparse metadata.
4. Extend `PrioritizedTaskPool.load_batch_to_runtime` to accumulate pending requests each step.
5. Land paged KV in static batching mode first; enable dynamic admission after that is green.

### Phase 3 — Modernize speculative decoding (in-place, not rip-and-replace)

User clarified 2026-04: SD is an **algorithmic** gap (vLLM 2.x-era), not a files-to-delete problem. Keep the scaffolding — session mgmt, RPC metadata, draft-target plumbing. Surgically upgrade the algorithm inside. See `SPEC_DECODING_SURVEY.md` for full arxiv-cited survey.

**Primary target: Sequoia-style external drafter + SpecInfer-style tree-attention verify.**
Why not EAGLE-2/3 or Medusa/Hydra: their drafters consume the target's near-final hidden state. In a Petals-style pipeline the tail peer owns the final layers, so each spec step would roundtrip. Sequoia's drafter is external and only needs target logits — ships through pipeline like normal traffic.

**Secondary / backup: Lookahead decoding** (no drafter, N-gram pool, self-speculative). Strong for repetitive/code generation. Cheap to keep as a second mode.

Surgical edits (all in place, no file deletions):
- `spe_dec_tree.py:170-185` — dense O(n²) ancestor matrix → packed block-lower-triangular mask.
- `speculative_model.py:619-631` — argmax sequential verify → vectorized probabilistic rejection sampling `min(1, p/q)`.
- `speculative_model.py:364-388` — coarse trim rollback → per-page rollback (depends on Phase 2 paged KV).
- `spec_decoding_drafter.py` — add dynamic tree shape selection (EAGLE-2 confidence-based pruning OR pre-computed Sequoia DP shape per GPU+model).
- Drop `session_max_length=624` hardcode (`speculative_model.py:58`).

EAGLE-2 deferred: revisit after Phase 2 ships and only if we add a "centralize speculation at tail peer" mode.

## V100 validation plan (admin@192.168.31.118 / 114514)

| Phase | V100 gate |
|-------|-----------|
| 0 | Correctness parity + decode tokens/s improvement |
| 1 | Zero regression, API refactor only |
| 2 | Continuous batching works under constrained concurrency |
| 3 | Functional smoke test only — NOT a perf gate |

LLaMA-7B fp16 ≈ 13 GB. 3 GB for KV + activations + paged metadata is tight. Drop spec decoding from the 16 GB perf bar; validate spec on a larger GPU.

## Top 3 rewrite risks

1. **Distributed protocol assumes contiguous per-request KV.** If remote peers encode a linear KV region in the wire format, Phase 2 becomes a protocol change, not a local refactor. Audit before Phase 2 implementation.
2. **Rollback / cache ownership underspecified today.** Both continuous batching and speculation need exact answers for logical length, accepted prefix, reclaim timing. Write invariants before coding.
3. **Mask/position semantics entangled with request lifetime** (`backend.py:324-378`). This is the core incompatibility; if positions aren't turned into per-token metadata + per-seq logical lengths early, Phase 2 bleeds into every layer boundary.
