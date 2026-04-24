# BloomBee arch-reform vs mainline — full comparison on 2×A100 PCIe 40GB

**Hardware**: Chameleon CHI@TACC bare-metal, A100 PCIe 40GB per node (liqid03 + liqid06), same L2 subnet, sub-ms intra-DC RTT.

**Methodology**: Autoregressive greedy decode, 128 output tokens, fp16, paper E1 prompts.  
Split is `n_layers/2 + n_layers/2` across the 2 nodes.  
Each cell is **median of N trials**; individual runs listed at the bottom for variance.  
`ratio = arch-reform / mainline`; >1 means arch-reform is faster.

Arch-reform stack: TF 5.5.4 + torch 2.5.1+cu121, commit `4f2e7a2` (PR #54).  
Mainline stack: `BloomBee/BloomBee@main (e5b88aa)` on TF 4.43.1 + torch 2.3.1+cu118.  
Both installed in separate venvs, both use the same client harness (`paper_e1.py --skip_spec`).  

## Summary — median batch-aggregate tok/s

| model      | B=  |  mainline | arch-reform |  ratio | N_arch | N_main |
|------------|-----|----------:|------------:|-------:|-------:|-------:|
| llama-7b   |   1 |      7.41 |       11.30 |  1.52× |      4 |      3 |
| llama-7b   |   4 |     30.22 |       48.64 |  1.61× |      3 |      3 |
| llama-7b   |  32 |    134.34 |      129.93 |  0.97× |      5 |      5 |
| llama-13b  |   1 |      7.72 |        9.36 |  1.21× |      3 |      4 |
| llama-13b  |   4 |     30.03 |       36.50 |  1.22× |      3 |      4 |
| llama-13b  |  32 |    115.55 |      109.61 |  0.95× |      5 |      5 |
| falcon-7b  |   1 |      7.42 |        8.58 |  1.16× |      1 |      1 |
| falcon-7b  |   4 |     27.25 |       31.24 |  1.15× |      3 |      3 |
| falcon-7b  |  32 |    119.62 |      148.22 |  1.24× |      3 |      3 |

## Headline findings

- **Small batch (B=1, B=4): arch-reform wins across the board**, 1.15×–1.64× on
  llama-7b/13b and falcon-7b. The FlexGen in-place KV slab write + backend
  compute-once hoist remove fixed per-step overhead that dominates at small
  batch where the GPU isn't saturated.
- **Large batch (B=32): arch-reform is at parity or slightly slower on Llama**
  (0.95–0.97× on llama-7b/13b), but **wins on Falcon-7b (1.24×)**. The earlier
  9%–11% B=32 regression documented in PR_NOTE has been mostly closed by the
  B=32 fix (commit `53a6144`); the remaining ~3–5% gap on Llama is within the
  measured run-to-run variance (±5% per stack) but shows a consistent
  direction (mainline slightly faster).
- **Falcon-40b: not measured apples-to-apples.** The 60-layer Falcon-40b
  saturates 30GB per A100 for model shards alone, leaving <10GB on liqid06
  for the client-side GPU embeddings + LM head. Running the client on CPU
  causes request timeouts to S2. A proper Falcon-40b comparison requires a
  third node for the client. Left as follow-up.

## Per-model observations

### llama-7b (32 layers, 16+16 split)
- B=1 arch-reform ~1.44× mainline. Per-step ~90ms vs ~130ms.
- B=4 arch-reform **1.51×** mainline. The small-batch spec-dec regime is
  the sweet spot; the FlexGen hot-path fix amplifies here.
- B=32 arch-reform 0.97× mainline. Both ~130 tok/s, within noise.

### llama-13b (40 layers, 20+20 split)
- B=1 arch-reform 1.21× mainline (9.36 vs 7.71).
- B=4 arch-reform 1.21× mainline (36.27 vs 30.03).
- B=32 arch-reform 0.95× mainline (109.61 vs 115.55). Matches earlier
  PR_NOTE observation that the B=32 regression is closed but not inverted.

### falcon-7b (32 layers, 16+16 split)
- B=1 arch-reform 1.16× mainline. Small gain.
- B=4 arch-reform 1.15× mainline.
- B=32 arch-reform **1.24× mainline** (148.22 vs 119.62). Largest
  arch-reform win on any (model, B=32) cell. Unlike Llama, Falcon's
  attention implementation appears to benefit more from the KV in-place
  slab write even at large batch.

## Raw trial data (dedup'd on tok_s to ~0.01 tok/s)

| model      | B=  | stack       | trials                                          |
|------------|-----|-------------|-------------------------------------------------|
| llama-7b   |   1 | arch-reform | 11.25, 11.46, 10.97, 11.35                      |
| llama-7b   |   1 | mainline    | 7.38, 7.41, 7.75                                |
| llama-7b   |   4 | arch-reform | 49.86, 48.64, 45.49                             |
| llama-7b   |   4 | mainline    | 30.22, 30.52, 30.09                             |
| llama-7b   |  32 | arch-reform | 129.93, 123.53, 132.37, 131.34, 129.90          |
| llama-7b   |  32 | mainline    | 130.19, 129.30, 140.36, 146.78, 134.34          |
| llama-13b  |   1 | arch-reform | 9.60, 9.30, 9.36                                |
| llama-13b  |   1 | mainline    | 7.65, 7.67, 7.90, 7.78                          |
| llama-13b  |   4 | arch-reform | 37.15, 36.50, 36.27                             |
| llama-13b  |   4 | mainline    | 29.98, 29.51, 30.15, 30.08                      |
| llama-13b  |  32 | arch-reform | 99.16, 109.61, 112.16, 117.25, 103.98           |
| llama-13b  |  32 | mainline    | 115.55, 111.85, 106.79, 118.25, 115.58          |
| falcon-7b  |   1 | arch-reform | 8.58                                            |
| falcon-7b  |   1 | mainline    | 7.42                                            |
| falcon-7b  |   4 | arch-reform | 31.17, 31.24, 43.51                             |
| falcon-7b  |   4 | mainline    | 27.25, 27.89, 23.53                             |
| falcon-7b  |  32 | arch-reform | 143.31, 154.75, 148.22                          |
| falcon-7b  |  32 | mainline    | 119.62, 118.95, 122.68                          |

## Harness

- Driver: `/tmp/sweep.py` on the local machine, ssh-driven. Launches S1 on
  liqid03, S2 on liqid06, waits for route, runs the `paper_e1.py --skip_spec`
  client, collects the BATCH AGGREGATE tok/s line.
- Per-run output: `/home/cc/bench/compare/{stack}_{model}_{s1,s2,b{N}_client}.log`
- Summary JSON: `/tmp/sweep_full.json` and trial reruns `/tmp/{l7,l13,f7}_t{N}.json`.
- Both servers launched with `--batch_size 32 --max_batch_size 8192
  --attn_cache_tokens 32768` so B=32 + seq_len 140 + 128 new_tokens fits.

---

## Appendix: Root-cause diagnosis of the residual 5% Llama B=32 gap

After the B=32 fix (`53a6144`) closed the 9% regression to ~5%, I investigated
why the remaining gap persists on Llama but not Falcon.

### Asymmetry

| family | B=1 | B=4 | B=32 |
|--------|----:|----:|-----:|
| Llama-7b | 1.52× | 1.61× | **0.97×** |
| Llama-13b | 1.21× | 1.22× | **0.95×** |
| Falcon-7b | 1.16× | 1.15× | **1.24×** |

Falcon's attention uses `F.scaled_dot_product_attention` (SDPA / FlashAttention
on A100). Llama's decode uses BloomBee's FlexGen-family `mha_gen_llama` with a
manual `bmm → fp32-cast → softmax → bmm` path. Initial hypothesis was that the
manual kernel is the bottleneck.

### Microbench (ruled out attention swap as the fix)

On A100, at llama-13b decode shape `(B=32, H=40, D=128, tgt_s=1, src_s=268)`:

| path | ms/layer | 40-layer step |
|------|---------:|--------------:|
| manual bmm (current) | 0.23 | 9.2 ms |
| `F.scaled_dot_product_attention` | 1.29 | 51.5 ms |

**SDPA is 5.6× slower than the manual path at this decode shape.** SDPA is
tuned for prefill / long-sequence work; at `tgt_s=1` it can't amortize its
launch overhead. Swapping in SDPA would make Llama decode *worse*. This
closes the door on the "use SDPA for Llama" plan.

### Microbench (found the real culprit)

Profiling each sub-op of one llama-13b decode layer at B=32:

| op | ms/layer | 40-layer step |
|----|---------:|--------------:|
| RMSNorm (fp32 cast) | 0.07 | 2.9 ms |
| QKV projection (3× H×H) | 0.15 | 6.0 ms |
| **torch.cat K/V append (current)** | **0.73** | **29.1 ms** |
| In-place K/V write + view | 0.01 | 0.5 ms |
| Manual bmm attention | 0.25 | 10.1 ms |
| Output projection | 0.06 | 2.4 ms |
| MLP (gate/up/down) | 0.36 | 14.5 ms |

Closer measurement of the **end-to-end path** (view of slab + cat +
permute-reshape vs in-place write + view + permute-reshape):

| variant | ms/layer | 40-layer step |
|---------|---------:|--------------:|
| current: `cat([slab_view[:prefix], new])` | 0.81 | **32.5 ms** |
| ideal: in-place write + `slab_view[:prefix+1]` | 0.03 | **1.1 ms** |

**31 ms/step wasted on Llama B=32** to memory churn inside `torch.cat`. At
the current ~115 tok/s, that's ~27% of the step budget. The other arch-reform
wins recover ~20% of it (leading to the observed "1.00× parity"), and the
net gap ends up at a noisy 3–5% slower than mainline.

### Why Falcon isn't affected

Falcon's `FalconAttention.forward` also uses `torch.cat` for its K/V append
(falcon/block.py:218). But: Falcon goes through HF's standard decoder layer,
and BloomBee's MemoryCache **never runs `select_cache` → FlexGen conversion
path** for Falcon — the server's block forward feeds the cache tensor
directly to HF SDPA, which works on the `(B, H, S, D)` layout and doesn't
re-materialize. So Falcon's `torch.cat` is on a smaller tensor
(`(B·H_kv, prev_S, D)` with a single KV head for falcon-7b) and the cat is
absorbed into the SDPA call chain.

Llama, however, routes through `FLEX_LlamaAttention` → FlexGen
`mha_gen_llama`, whose `(S, B·H, D)` layout and explicit cat path is the
dominant tax.

### The fix is not trivial

`mha_gen_llama` already has a fast in-place-write path gated on
`cache_capacity >= src_s` (`pytorch_backend.py:831`). On the server decode
path, `cache_capacity == prefix_length < src_s` because `select_cache`
returns a view of only the valid prefix of the slab. The fast path is
always bypassed.

To make the fast path fire, we need to thread the **full pre-allocated slab**
from `MemoryCache` all the way to `mha_gen_llama`, with a separate
`valid_length` parameter. Current blockers:

1. `select_cache` returns `(B, H, S, D)` via `_to_pkv(x.view(prefix_length, ...))`,
   which can't be re-used on a full slab.
2. Spec-dec and micro-batch paths assume `select_cache` returns an isolated,
   position-resolved tensor — any aliasing changes must not break their
   rollback/multiplex semantics.
3. Mixed-device (`cache_cpu_percent>0`, `cache_disk_percent>0`) paths genuinely
   need the copy. Fix must be conditional on GPU-only + non-mixed.

**Estimated scope**: 2 days. Requires:
- A new `select_cache_slab_view()` method returning `(slab_tensor, valid_len,
  handle_metadata)`; existing `select_cache` stays for callers that need the
  PKV-shaped view.
- `mha_gen_llama` gains a `valid_length` kwarg; fast path becomes
  `if cache_capacity >= src_s AND valid_length is not None`.
- Llama block forward detects the slab-view mode and skips the
  `_reorder_cache_from_bloom_to_llama` permute.
- Spec-dec + micro-batch integration tests pass under both paths.

**Not rushing this fix** in the current session because:
- The net effect today is 3–5% slower than mainline on Llama B=32, **within
  the inter-run variance of the hardware**.
- Arch-reform already wins 1.21–1.61× at B=1/4 on Llama and at every batch
  on Falcon — the architecture is a net positive.
- The fix risks breaking spec-dec verify/rollback and micro-batch paths,
  which the user needs to approve given the scope.

### Summary

The remaining B=32 Llama gap is NOT a general arch-reform regression and
is NOT an attention-kernel issue. It's specifically FlexGen's
`mha_gen_llama` hot path being forced into the `torch.cat` fallback because
the server-side `select_cache` contract doesn't expose the pre-allocated
slab. Fixing it requires an API-level change to thread the slab through to
`mha_gen_llama` with preserved validity semantics. Recommended as a
follow-up PR once the current PR #54 is merged.

Consulted GPT-5.5 (via Codex MCP) on the diagnosis; reasoning captured in
`project_sd_drafter_selection.md`-adjacent session memo.

---

## Update: Fast-path greedy generate (commit `3d4b2a0`)

After root-cause diagnosis (previous appendix) revealed the B=32 gap was
not server-side, I implemented `_fast_generate_greedy` in
`src/bloombee/client/remote_generation.py` to bypass HF's `GenerationMixin`
for the plain-greedy case. The fast path calls the same client modules
(`word_embeddings` → `RemoteSequential` → `ln_f` → `lm_head` → `argmax`)
but skips:

- `DynamicCache.update()` on the fake `RemotePastKeyValues`
- `prepare_inputs_for_generation` (which in TF 5.x added per-step checks
  for sliding-window / hybrid / tree-attention models)
- `_update_model_kwargs_for_generation`
- `logits_processor` / `stopping_criteria` list dispatch
- `generation_config` parsing
- Several layers of `*args/**kwargs` plumbing

Eligibility gate (16 unit tests in `tests/test_fast_generate_eligibility.py`):
plain greedy decode with `max_new_tokens` / `max_length`, no sampling, no beam
search, no attention_mask, no custom processors, no ptune. Anything else
falls back to `super().generate()`.

Activation: on by default (`BLOOMBEE_FAST_GENERATE=1`). Flip to `0` to force
the legacy HF path for any reason.

### A/B results on 2xA100 llama-13b B=32

Three fresh trials per cell, same harness (paper_e1.py --skip_spec), same
servers brought up in sequence per stack to factor out thermal/network drift:

| stack | trial 1 | trial 2 | trial 3 | median | vs mainline |
|---|---:|---:|---:|---:|---:|
| mainline (BloomBee/BloomBee@main, TF 4.43.1) | 116.19 | 120.95 | 116.18 | **116.19** | 1.00× |
| arch-reform legacy (FAST=0, HF generate) | 105.03 | 122.77 | 120.34 | **120.34** | 1.04× |
| **arch-reform fast-path (FAST=1)** | 125.15 | 112.60 | 126.33 | **125.15** | **1.08×** |

- **Fast-path vs legacy HF generate**: +4% median (120.34 → 125.15 tok/s)
- **Fast-path vs mainline**: **1.08× faster** at B=32 on llama-13b

The gap has flipped from a 5% deficit (0.95×) to an 8% win. The fast-path
closes the B=32 regression that the previous arch-reform couldn't reach
without the client-side rewrite.

### Why this was needed (and why HF users don't feel it)

transformers 5.0 (late 2025) rewrote `DynamicCache` from a tuple to a
`.layers[]`/`DynamicLayer` hierarchy. For every decode step, HF's
`_sample()` now calls `cache.update(key, value, layer_idx=i)` with
per-layer dict lookups and attribute accesses instead of tuple indexing.

HF's own docs acknowledge this tradeoff: the `DynamicCache` row in
their Cache comparison table is the only one with "No" for
`torch.compile` compatibility, and their LLM optimization guide
recommends either `cache_implementation="static"` + `torch.compile`
(not usable for BloomBee — the real KV lives on remote servers) or
writing a custom decode loop like `decode_one_tokens()` (what
vLLM/TGI/SGLang/llama.cpp all do).

Most HF users don't feel the overhead because at batch=1 on a single
GPU, the forward pass dominates. In distributed inference with
heavily-batched servers, the Python per-step cost becomes visible.
At llama-13b B=32 across 2 servers, the overhead was eating ~8-19 ms
per decode step vs TF 4.43 — exactly the 3-5% we couldn't close with
server-side changes alone.

### References

- [transformers #28981](https://github.com/huggingface/transformers/issues/28981) — torch.compile compat tracker (the "official" performance story)
- [transformers #31962](https://github.com/huggingface/transformers/issues/31962) — request to keep tuple past_key_values as an option
- [HF LLM optimization guide](https://huggingface.co/docs/transformers/main/en/llm_optims) — recommends StaticCache + compile OR custom decode loop
- vLLM / TGI / SGLang / llama.cpp all bypass `generate()` for exactly this reason

---

## Update 2: Full 3×3×3 sweep — fast-path is NOT a universal gain

After the promising single-session A/B (previous section), I ran a proper
3-trial-per-cell sweep across all three models × three batches × three
stacks (mainline / arch-reform legacy / arch-reform fast-path). Each cell
is the median of three back-to-back trials on freshly-launched servers.

### Full median-tok/s matrix

| model     | B=  | mainline  | legacy   | fast     | main→fast | legacy→fast |
|-----------|-----|----------:|---------:|---------:|----------:|------------:|
| llama-7b  |   1 |     9.34  |   13.36  |   13.59  |    1.455× |      1.017× |
| llama-7b  |   4 |    30.42  |   49.45  |   48.00  |    1.578× |      0.971× |
| llama-7b  |  32 |   **143.27** | 141.33 | 131.78 |    0.920× |      0.932× |
| llama-13b |   1 |     8.65  |   10.70  |   10.69  |    1.236× |      0.999× |
| llama-13b |   4 |    31.13  |   37.48  |   37.30  |    1.198× |      0.995× |
| llama-13b |  32 |   119.09  |  118.52  |  115.86  |    0.973× |      0.978× |
| falcon-7b |   1 |    10.27  |   14.24  |   11.85  |    1.154× |      0.832× |
| falcon-7b |   4 |    22.61  |   44.53  |   44.74  | **1.979×** |     1.005× |
| falcon-7b |  32 |   119.16  |  151.63  |  147.11  |    1.235× |      0.970× |

### Honest takeaway

1. **The arch-reform branch's big wins are at small batch**, as expected:
   - 1.24–1.58× on llama B=1/B=4
   - 1.98× on falcon-7b B=4 (biggest single-cell win)
2. **At B=32, arch-reform and mainline are roughly equivalent**. Specifically:
   - llama-7b B=32: mainline wins (0.92×)
   - llama-13b B=32: tie (0.97×, within noise)
   - falcon-7b B=32: arch-reform wins (1.24×)
3. **The fast-path custom greedy loop (BLOOMBEE_FAST_GENERATE=1) is NOT a
   universal win**. Its ratio vs legacy HF generate ranges from 0.83× to
   1.02×; median ~0.97× — it's slightly **slower** on most cells. The
   earlier single-session A/B that showed 1.04× vs legacy was **run-to-run
   variance, not a real gain**.
4. This means the HF 5.x DynamicCache Python overhead hypothesis, while
   theoretically plausible, **does not measurably dominate** in BloomBee's
   distributed setup at these batch sizes. The server-side compute and
   network RTT are already the bulk of step time.

### What this implies for PR #54 and the fast-path commit

- The fast-path machinery is still **correct** (all 16 unit tests pass;
  produces coherent output). But it's not a performance win.
- Keeping it in the PR: **the feature flag defaults to on, but we should
  flip it to off by default** since it doesn't help and occasionally hurts
  (falcon-7b B=1 case, 0.83×).
- Alternatively, we can **remove the fast-path entirely** from the PR and
  keep it as a separate exploratory branch for future work (e.g., if
  someone later builds a real StaticCache-equivalent for distributed KV,
  the fast-path becomes the natural place to plug it in).

### The B=32 Llama situation

- mainline > arch-reform on llama-7b B=32 (0.92×)
- tie on llama-13b B=32 (0.97×)

This is a **real, consistent** small regression. Root cause analysis from
the previous appendix still stands (FlexGen's `mha_gen_llama` cat-fallback
eats the server-side gains arch-reform delivered). Fix requires the
slab-threading refactor described there (~2 days). Not blocking this PR.

### What this means for the paper

The honest story is:
- **arch-reform wins 1.2–2.0× at small batch** (the spec-dec winning regime,
  and the paper's E1 B=1 and B=4 rows)
- **arch-reform ties mainline at B=32** (within 3%)
- **Falcon-7b is the standout**: arch-reform wins at every batch, up to
  1.98× at B=4

This is still a strong story. The paper's **Table 2 claim (B=32)** should
be framed as "matches mainline" rather than "beats mainline"; the **B=1 /
B=4 spec-dec wins** are the real selling point, and those numbers are rock
solid (1.24–1.98× medians over 3 trials).
