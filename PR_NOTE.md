# arch-reform + Transformers 5.x migration

This branch rebases BloomBee onto HuggingFace Transformers 5.x
(`transformers>=5.5,<5.6`), adds Qwen3 as a first-class model family,
lands three of the four items from the arch-reform memo, and re-works
speculative decoding into an in-tree implementation of SpecInfer,
EAGLE-2, and Sequoia.

## What lands

| Area | Status |
|---|---|
| Transformers 5.x migration (Bloom / Falcon / Llama / Mixtral / Qwen3) | Landed |
| Qwen3 model family (first-class, with parity test) | Landed |
| Speculative decoding upgrade (SpecInfer + EAGLE-2 + Sequoia) | Landed |
| Backend compute-once hoist | Landed |
| FlexGen KV in-place slab write | Landed |
| B=32 AR regression fix (paged-KV gate + vectorized mask) | Landed |
| Continuous batching | Separate branch |

## Benchmarks — 2 × A100 PCIe 40 GB, llama-13b, 20+20 split, fp16, greedy, 128 output tokens

Same hardware, same prompts, same split. Mainline is
`BloomBee/BloomBee@main` on its own TF 4.43.1 + torch 2.3.1 venv;
arch-reform is this branch on TF 5.5.4 + torch 2.8.0.

**Autoregressive (no spec-dec)**:

| B  | mainline (batch-agg tok/s) | arch-reform | ratio |
|---:|---:|---:|---:|
| 1  | 4.14 (llama-7b) | 4.70 (llama-7b) | 1.14× |
| 4  | 27.93 | **33.18** | **1.19×** |
| 32 | 115.79 | **116.00** | 1.00× |

- B=1/B=4: the fixed per-step overhead arch-reform removes (KV in-place
  slab write + backend hoist) dominates on light workloads.
- B=32: matches mainline within noise after the
  2026-04-23 fix (commit `53a6144`). Before that fix the ratio was
  0.91× — the paged-KV shim was paying per-step dict ops + the causal
  mask was being rebuilt via a Python loop every step. Fix details:
  - `BLOOMBEE_PAGED_KV` gate on `_track_paged_write` /
    `_rollback_paged_to` / `_commit_paged_to` so the shim is truly
    zero-cost when off.
  - `_create_causal_attention_mask` replaced the `for i in
    range(current_token_count)` loop with a single
    `(cols <= rows)` comparison. Parity verified across all shapes.
  - Decode-path scores tensor (all-zeros on AR decode) cached per
    `(batch_size, src_len, device)`.

**Speculative decoding at B=4** (arch-reform only; mainline can't run
Sequoia's `List[int]` widths):

| variant | per-seq tok/s | batch-agg tok/s | vs arch-reform AR |
|---|---:|---:|---:|
| AR | 8.04 | 32.16 | 1.00× |
| spec-dec fixed-grid d=4 w=2 | 9.67 | 38.69 | 1.20× |
| Sequoia [8] | 8.26 | 33.05 | 1.03× |

**Spec-dec at B=32**: slower than AR (0.83× for mainline d=4 w=2,
0.76× for arch-reform d=4 w=2) — expected; the paper's E1 row shows
the same shape for LLaMA-30B at B=32.

**Qwen3-14B reference** (2 × A100, B=1, fp16, greedy): 110 ms/step,
9.09 tok/s, coherent output. No mainline comparison — mainline pins
TF <4.44 and does not support Qwen3.

## Architecture changes

### Transformers 5.x compat

1. **Cache API** — attention now takes `Cache` objects with `.update()`
   instead of raw `(k, v)` tuples. BloomBee still routes state through
   its own `MemoryCache` slab; small per-model `OptimizedXxxAttention`
   subclasses bridge the two.
2. **tie_weights** — upstream walks `_tied_weights_keys` via
   `get_parameter`, which trips on BloomBee's `LMHead.weight = None`
   placeholder. Uniform `mark_tied_weights_as_initialized` no-op +
   manual tie.
3. **from_pretrained device context** — TF 5.x wraps `__init__` in
   `torch.device('cuda')`, which hijacks hivemind's CPU tensor setup.
   Each `DistributedXxxModel.__init__` wraps `RemoteSequential`
   construction in `with torch.device('cpu'):`.
4. **prepare_inputs_for_generation** — cache-aware reimplementation on
   Falcon and Qwen3 mirroring Llama; the upstream default crashes on
   `RemotePastKeyValues`.
5. **Local weight loading** — `hf_compat.get_file_from_repo` shim for
   TF 5.x's changed `get_file_from_repo` behavior.

### arch-reform hot-path

- **Backend hoist** (`server/backend.py`): `_flag_to_bool` promoted to
  module scope; `max(shard_num_heads)` and the 5-field cache-reuse
  policy gate cached in `__init__`. No API change.
- **FlexGen KV in-place slab write**
  (`flexgen_utils/pytorch_backend.py`): when the cache capacity covers
  the target sequence length, write new tokens in place instead of
  `torch.cat`. Falls back to concat otherwise (e.g. server's
  sliced-per-step cache).
- **Paged-KV primitive** (`server/paged_kv.py`): additive side-channel,
  opt-in via `BLOOMBEE_PAGED_KV=1`. Tested but not load-bearing on the
  default hot path.

### Spec-dec upgrade

- EAGLE-2 dynamic draft tree (`tree_budget`, `tree_min_log_prob`).
- SpecInfer rejection sampling for `do_sample=True` (new capability;
  mainline hard-codes `do_sample=False`).
- Sequoia static per-depth plan (`beam_width=List[int]`) with DP over
  an online acceptance histogram.
- Default behavior (fixed-grid greedy) is byte-identical to pre-change.

### Qwen3 family

Mirrors the existing Bloom/Falcon/Llama layout under
`src/bloombee/models/qwen3/`. Block-level parity test against the HF
reference block is included.

## Test plan

- [x] E2E greedy on llama-13b (2 × A100) — mainline vs arch-reform AR at
      B=4 and B=32 (see table above)
- [x] Qwen3-14B 2-server coherent output on A100
- [x] Regression check: llama-7b, qwen3-0.6B, bloom-560m, falcon-rw-1b
      E2E on V100 under TF 5.5.4
- [x] Block parity tests: qwen3 vs HF reference (float16 tolerance)
- [x] Paged-KV rollback unit tests (10 tests, in `tests/`)
- [ ] Continuous batching — excluded from this PR, separate branch
- [ ] Client-side GPU LMHead for Qwen3-14B — follow-up commit
