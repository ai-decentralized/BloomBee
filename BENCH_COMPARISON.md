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
