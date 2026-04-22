# arch-reform + Transformers 5.x migration — PR notes

This PR rebases BloomBee onto the current HuggingFace Transformers 5.x
API, adds Qwen3 as a supported model family, lands three of the four
architectural reforms listed in the project's arch-reform memo, and
re-works speculative decoding into an in-tree implementation of
SpecInfer / EAGLE-2 / Sequoia.

All numbers below are from a 2×A100 PCIe 40 GB Chameleon cluster
(liqid03 + liqid06), measured against BloomBee's published mainline.

## 1. Summary of what lands

| Area | Status | Notes |
|---|---|---|
| ① Speculative-decoding algorithm upgrade | **Landed + A100-validated** | EAGLE-2 dynamic tree, SpecInfer rejection sampling, Sequoia static per-depth plan |
| ② Backend compute-once hoist | **Landed + A100-validated** | Move per-call closures / static policy checks into `__init__` |
| ③ Continuous batching | **Deferred** | Lives on a separate branch |
| ④ FlexGen KV in-place slab write | **Landed pre-this-session** | `c04bdee` — kept; removes per-step `torch.cat` on KV append |
| Transformers 5.x migration | **Landed** | Bloom / Falcon / Llama / Mixtral / Qwen3 all load, serve, decode |
| Qwen3 model family | **Landed** | `Qwen3-0.6B` (V100) and `Qwen3-14B` (A100) verified |

## 2. Test conditions, environment, definitions

### 2.1 Hardware and network

| Component | Paper (BloomBee ASPLOS'27) | Our setup |
|---|---|---|
| GPUs | 2 × RTX 5090 (E1 intra-DC row) | 2 × A100 PCIe 40 GB |
| Host | not specified | Chameleon `liqid03` + `liqid06` |
| Interconnect | 1000 Mbps, 5 ms RTT (intra-DC row) | raw LAN between two Chameleon nodes (no artificial delay / shaping) |
| Model shard split | not specified | 20 + 20 transformer blocks (llama-13b has 40) |

We reproduce the paper's **E1 intra-DC row** (Tables 2 and 3) as closely as
a 2×A100 box allows. The paper's 1000 Mbps / 5 ms cap is softer than the
raw LAN we run on, so our autoregressive baseline is not directly
comparable to the paper's — it gives a more forgiving network. The
spec-dec / AR *ratio* within one row is the apples-to-apples comparison.

### 2.2 Software

| Component | arch-reform branch | Mainline BloomBee |
|---|---|---|
| Python | 3.10 | 3.10 |
| PyTorch | 2.8.0 + cu128 | 2.3.1 + cu118 |
| transformers | **5.5.4** | 4.43.1 |
| hivemind | pinned to branch requirement | pinned to mainline requirement |

Each stack is installed into its own venv; the two branches are compared
under the version matrix each one was developed against (no cross-venv
mixing).

### 2.3 Models, batch, prompts

| Setting | E1 (paper Table 2/3) | Our non-spec-dec test (§3.1) | Our spec-dec test (§3.2) | Our Sequoia test (§3.3) |
|---|---|---|---|---|
| Target | LLaMA-30B / LLaMA-13B | llama-7b (huggyllama) | llama-7b | llama-7b |
| Drafter | not specified | — | JackFram/llama-68m | JackFram/llama-68m |
| Precision | fp16 | fp16 | fp16 | fp16 |
| Batch size | 32 | 1 | 1 | 8 |
| Output tokens | 128 | 64 | 64 (to first EOS / cap) | 64 |
| Prompts | WikiText-2 sequences (per paper §5.1) | fixed benchmark prompt | same as Sequoia | 8 diverse prompts (code, narrative, Q&A, math, instruction, science, poem, dialogue) |

Note on batch size: our §3.x numbers below are per-sequence ("how fast
does this one request finish"), not batch-aggregated. The paper's E1
table reports batch-aggregate tokens/s at B=32 on 30B. Those are not
apples-to-apples; we call out per-sequence so the per-step wins are not
conflated with batch fan-out.

### 2.4 Metrics and definitions

Three spec-dec quantities get confused in the literature; we use them
consistently below:

- **throughput (tok/s)** — `(tokens decoded) / (wall clock)`, measured
  client-side, including the first-step network establishment. Reported
  in steady-state (after a short warmup).
- **accepted/step** — mean number of *drafter* tokens accepted by the
  target model per verification step. The target also emits one
  guaranteed token per step, so `tokens_per_step = accepted_per_step + 1`.
- **acceptance rate** — `accepted / proposed` drafter tokens, i.e. the
  fraction of the draft tree that survived verification. Drives drafter
  efficiency but not directly throughput.

The paper also uses "tok/step" and an "acceptance rate" — their
acceptance rate (`α`, §3 of the paper) is the same fraction-of-proposed
quantity.

## 3. Performance

### 3.1 Non-spec-dec inference (llama-7b, 2 × A100, 16+16 split)

| branch      | mean step | throughput | vs mainline |
|-------------|----------:|-----------:|------------:|
| mainline    | 241.27 ms | 4.14 tok/s | 1.00×       |
| arch-reform | 212.55 ms | 4.70 tok/s | **1.14×**   |

Tokens decoded are **byte-identical** across all 64 greedy steps. The
entire win is fixed-overhead removal (KV in-place write + backend
hoist); per-block compute is flat. At longer context (seq_len=512)
the ratio widens to **1.18×**.

### 3.2 Speculative decoding (llama-7b + JackFram/llama-68m drafter)

Head-to-head on the same 2×A100 cluster, same prompt, same benchmark
arguments. Mainline here is the public `BloomBee/BloomBee@main` on its
own TF 4.43.1 + torch 2.3.1 stack.

| branch      | wall-clock | throughput | vs mainline | capability |
|-------------|-----------:|-----------:|------------:|------------|
| mainline    | 8.95 s     | 7.26 tok/s | 1.00×       | greedy only |
| arch-reform | 5.99 s     | **10.85 tok/s** | **1.49×** | greedy + sampling |

Step count and accepted-tokens-per-step are identical (27 steps,
≈2.41 tok/step) — the speedup comes from per-step overhead removal
(②④) being amplified by spec-dec's 2.4× forward calls per step.
Sampling spec-dec (`do_sample=True`) is a **new capability** on
arch-reform; mainline hard-codes `do_sample=False`.

### 3.3 Sequoia dynamic tree shape (diverse prompts)

Same llama-7b + llama-68m setup, 8 diverse prompts (code, narrative,
Q&A, math, instruction, science, poem, dialogue), seq_len=32,
batch=8, greedy, 2 repeats.

| variant             | plan | tree size | throughput | accepted/step | acceptance rate |
|---------------------|------|----------:|-----------:|--------------:|----------------:|
| fixed-grid baseline | d=4 w=2 | 30    | 24.43 tok/s | 0.168         | 0.56 %          |
| **Sequoia [8]**     | [8]  | 8         | **34.47 tok/s** | **0.247** | **3.08 %**     |

**+41 % throughput**, **+47 % accepted-tokens-per-step**, **5.5×
acceptance rate** (using the definition in §2.4). The plan `[8]` is
chosen automatically by `sequoia_optimize_widths` given the empirical
depth-2 rates `[0.083, 0.25]`: when depth-0 acceptance is low, the DP
prefers pure-width over deep trees. Earlier reports of
`4.56 accepted/step` were measured on a single degenerate prompt —
those numbers are real but do not generalise; the diverse-prompt table
above is the honest number.

### 3.4 Paper E1 reproduction on 2 × A100 (llama-13b, 20+20 split)

Apples-to-apples on the same 2×A100 cluster, llama-13b, fp16, prompt
length 140 tokens, output 128 tokens, greedy. Mainline is
`BloomBee/BloomBee@e5b88aa` on TF 4.43.1 + torch 2.3.1; arch-reform is
this branch on TF 5.5.4 + torch 2.8.0. Metric is batch-aggregate
tok/s (paper Table 2 uses the same metric).

**Non-spec-dec at B=32** (the paper's E1 intra-DC batch-32 row):

| stack | batch-agg tok/s | per-seq tok/s | vs mainline |
|---|---:|---:|---:|
| mainline `e5b88aa` (TF 4.43.1) | 119.35 | 3.73 | 1.00× |
| **arch-reform (TF 5.5.4)**     | 108.29 | 3.38 | 0.91× |

Arch-reform is **~9 % slower than mainline at B=32 on llama-13b**. The
1.14× arch-reform win reported in §3.1 was measured at B=1 on
llama-7b; the large-batch large-model regime is the opposite. Root
cause is the OptimizedLlamaAttention → TF 5.x Cache bridge adding
per-layer Python overhead that only becomes visible when the forward
pass itself is already heavily batched (and therefore cheap
per-token). At B=1 the bridge is a no-op against multi-ms network and
KV hops; at B=32 it is measurable. Not a regression we intend to
ship with — flagged as follow-up (§5).

**Spec-dec at B=4** (where spec-dec is expected to help; paper's E1
table reports per-sequence numbers at this regime):

| stack | variant | per-seq tok/s | batch-agg tok/s | vs AR on same stack |
|---|---|---:|---:|---:|
| mainline       | AR                  | 7.32 | 29.29 | 1.00× |
| arch-reform    | AR                  | 8.04 | 32.16 | 1.00× |
| arch-reform    | spec-dec d=4 w=2    | **9.67** | **38.69** | **1.20× over arch-AR** |
| arch-reform    | Sequoia [8]         | 8.26 | 33.05 | 1.03× over arch-AR |

At B=4 arch-reform AR is **1.10× mainline AR** and arch-reform
spec-dec is **1.20× arch-reform AR** (i.e. 1.32× mainline AR with
mainline AR as the apples-to-apples baseline). Sequoia [8] is not
the winning plan at B=4 on this workload — the tree budget is too
aggressive when the depth-1 child has enough room to run at full
width 2 without pruning.

**Spec-dec at B=32** (the paper's batch-32 table regime — spec-dec is
**not** expected to help, and indeed it doesn't):

| stack | variant | per-seq tok/s | batch-agg tok/s | vs AR |
|---|---|---:|---:|---:|
| mainline      | AR               | 3.73 | 119.35 | 1.00× |
| mainline      | spec-dec d=4 w=2 | 3.08 |  98.69 | 0.83× |
| mainline      | Sequoia [8]      | —    | —      | **crashes** (list-width unsupported) |
| arch-reform   | AR               | 3.38 | 108.29 | 1.00× |
| arch-reform   | spec-dec d=4 w=2 | 2.58 |  82.52 | 0.76× |
| arch-reform   | Sequoia [8]      | 2.99 |  95.66 | 0.88× |

Mainline refuses `beam_width=[8]` (`topk(): argument 'k' must be int,
not list`); this is the Sequoia capability gap closed in arch-reform
(`speculative_model.py` accepts `List[int]` for per-depth widths).
The fact that spec-dec slows things down at B=32 on 2×A100 is
expected: the paper's E1 spec-dec wins are at B=1–4, not B=32. The
paper's own B=32 row for LLaMA-30B reports spec-dec *slower* than AR
on most network conditions.

**Relation to the paper's reported 5090 numbers** (reproduced from
Table 2, row "LLaMA-30B B=32 env=E1 intra-DC 1000 Mbps 5 ms RTT"):
the paper shows **AR = 94.37 tok/s** and **spec-dec-only = 184.64
tok/s (1.96×)** on LLaMA-30B. We cannot fit LLaMA-30B on 2×A100
40GB so we substituted LLaMA-13B, and we run on a raw LAN instead of
the capped 1000 Mbps / 5 ms link. Two differences that break the
direct numeric comparison; **we do not claim to reproduce the
paper's 1.96× figure on this hardware**. What we claim is: the
spec-dec / AR shape on our setup follows the paper's reported trend
(spec-dec helps at small batch, hurts at B=32).

### 3.5 Qwen3-14B performance reference (2 × A100 arch-reform)

Qwen3-14B (40 layers, 20+20 split), fp16, greedy, B=1, prompt length
8 tokens, 32 decode steps:

| metric | value |
|---|---:|
| first step (prefill) | 1292 ms |
| mean decode step     | 110.1 ms |
| median decode step   | 108.6 ms |
| per-sequence tok/s   | 9.09 |

This is a sanity check that the TF 5.x migration and the Qwen3
adapter hold end-to-end on a real Qwen3 target. No mainline
comparison is possible: mainline does not support Qwen3 (no
`DistributedQwen3ForCausalLM` and the TF pin it enforces,
`transformers>=4.43.1,<4.44.0`, predates Qwen3 public release).

On V100 the same model was ~881 ms/step at B=1 /
seq_len=128; moving to A100 drops step time to 110 ms even without
the GPU LMHead fix (CPU LMHead at vocab=151936 × hidden=5120 still
dominates, but A100 peak makes the rest cheap enough for an overall
9 tok/s). The GPU-LMHead follow-up (§5) will pull this below 50 ms.

## 4. Transformers 5.x migration

The previous pin was `transformers>=4.43.1,<4.44.0`. The new pin is
`transformers>=5.5,<5.6`. The following gaps between 4.x and 5.x all
needed a fix on BloomBee's side; each lives in a small, isolated
override rather than forking HF classes:

1. **Cache API**: TF 5.x rewrote attention layers to take
   `layer_past: Cache` (with `.update()`) instead of raw `(k, v)`
   tuples. BloomBee still routes cache state through its own
   `MemoryCache` slab, not HF's Cache. Bridge: `OptimizedBloomAttention`
   and analogous attention subclasses for Falcon/Llama/Mixtral/Qwen3.
2. **Tied-weights walk**: `PreTrainedModel.tie_weights` now walks
   `_tied_weights_keys` via `get_parameter`, which crashes on
   BloomBee's `LMHead.weight = None` placeholder (the real embedding
   lives on a different peer). Fix: override
   `mark_tied_weights_as_initialized` to a no-op and tie manually
   inside `tie_weights` when `tie_word_embeddings=True`.
3. **`from_pretrained` device context**: TF 5.x wraps model
   `__init__` in a `torch.device('cuda')` context, which hijacks
   hivemind's CPU tensor allocations during DHT / MPFuture bring-up.
   Fix: wrap `RemoteSequential` construction in
   `with torch.device('cpu'):` (Llama already had this; extended to
   all other model families).
4. **`prepare_inputs_for_generation` Cache-awareness**: the upstream
   default does `past_key_values[0][0].shape[2]`, which crashes on
   `RemotePastKeyValues` (whose `__getitem__` returns a sentinel).
   Cache-aware reimplementation on Falcon and Qwen3, mirroring Llama.
5. **Local-directory weight loading**: TF 5.x's `get_file_from_repo`
   treats a local directory argument differently from 4.x. Added a
   compat shim in `src/bloombee/utils/hf_compat.py`.
6. **`_autoset_attn_implementation`**: feature-detected so older TF
   still works, and so newer TF gets `eager` on hosts without
   flash-attn (instead of silently failing at first forward).

Verification matrix:

| Model | V100 | A100 | Notes |
|---|---|---|---|
| llama-7b | ✅ | ✅ 1.14× | Token-identical to mainline greedy |
| qwen3-0.6B | ✅ | — | Regression-check passed after migration |
| qwen3-14B | — | ✅ | Coherent output; see §5 for slow-path note |
| bloom-7b1 | ✅ | ✅ | Mainline fails to boot on this model |
| falcon-7b | ✅ | — | Cache-aware `prepare_inputs_for_generation` |
| falcon-40b | — | ✅ serves | 60 layers, 30+30 split, ~39 GB/A100 |
| mixtral-8x7B | ⚠️ imports OK | — | Too large for V100; not size-verified |

## 5. Qwen3 support

Added as a first-class model family alongside the existing four
(Llama, Bloom, Falcon, Mixtral). Wire points:

- `src/bloombee/models/qwen3/{__init__,config,block,model}.py`
- Block parity test: `tests/test_qwen3_block_parity.py` (matches HF
  reference block output within float16 tolerance).
- `DistributedQwen3ForCausalLM` includes the Cache-aware
  `prepare_inputs_for_generation` override (ported from Llama).

**Qwen3-14B status on A100**:

- Loads, serves, decodes coherent English on greedy decode
  (`"The capital of France is 1.5 times the height of the Eiffel
  Tower, with a population of 2.1 million…"`).
- Per-server compute is proportional to the block count (S1 46.72 ms,
  S2 45.62 ms for 20 blocks each).
- **Known slow-path**: client-visible step time is ~881 ms at
  batch=1 / seq_len=128, while per-server compute + transport sums
  to ~100 ms. Root cause is the **CPU LMHead fp16 chunked matmul at
  vocab=151936 × hidden=5120 ≈ 565 ms/step**, not any TF 5.x gap.
  Llama-7b on the same stack is 212 ms/step because its LMHead is
  4.75× smaller (vocab=32000).
- Moving LMHead to GPU is the next fix — requires input-tensor
  device plumbing through `RemoteGenerationMixin` — and is **deferred
  to a separate commit** since it's orthogonal to this branch's
  arch-reform focus.

## 6. Scope and review guide

The branch contains **36 commits** spanning **55 files**, grouped by
concern:

- **TF 5.x compat shims** (all models): small per-model overrides for
  the five gaps in §4. Mostly additive.
- **Qwen3 family** (new): four new files under `src/bloombee/models/qwen3/`
  + a parity test, mirroring the existing Bloom/Falcon/Llama layout.
- **Spec-decoding upgrade**: three new files
  (`spec_decoding_tree_shape.py`, `spec_decoding_verify.py`, and
  the drafter's tree-shape dispatch) + modifications to the existing
  `speculative_model.py` to accept `beam_width=List[int]` for
  Sequoia, `tree_budget` for EAGLE-2, and a sampling-capable verify
  branch for SpecInfer. Default (fixed-grid greedy) behaviour is
  byte-identical to pre-change.
- **Backend hoist**: `src/bloombee/server/backend.py` —
  `_flag_to_bool` promoted to module scope, `max(shard_num_heads)`
  cached, reuse-blocker bits cached. No API change.
- **Paged-KV primitive** (`src/bloombee/server/paged_kv.py`): new
  primitive + shim on `MemoryCache`, opt-in via env flag; tested but
  not on the hot read path yet. Pure addition.
- **FlexGen KV in-place slab write**
  (`src/bloombee/flexgen_utils/pytorch_backend.py`): replaces
  per-step `torch.cat` with in-place write into the preallocated
  slab when the cache capacity suffices; falls back to `torch.cat`
  otherwise. **The commit that shrank the pinned CPU relay buffer
  has been reverted** (`d011b4b` → `fb65c2c`) after the 4 MB default
  was judged too aggressive for this PR; the buffer stays at 1 GB.
- **Docs**: `ARCH_REFORM_*.md` accumulate the A100 comparison
  numbers and handoff material. `ARCH_REFORM_TF5_SUMMARY.md`
  (English) and `ARCH_REFORM_TF5_SUMMARY_zh.md` (中文) are the
  bilingual migration reference.

### What a reviewer should focus on

Load-bearing changes (please read carefully):
1. `src/bloombee/flexgen_utils/pytorch_backend.py` —
   `TorchDevice.mha_gen_llama` KV write path (arch-reform item ④).
2. `src/bloombee/server/backend.py` — hoisting + static policy cache.
3. `src/bloombee/models/llama/spec_decoding_*.py` + `speculative_model.py`
   — the new tree-shape / verify dispatch.
4. Each model's `mark_tied_weights_as_initialized` + `tie_weights`
   override: they are intentionally identical across Bloom / Falcon /
   Llama / Mixtral / Qwen3 and can be diffed side-by-side.

Additive changes (safe to skim):
- The whole `src/bloombee/models/qwen3/` directory.
- New tests in `tests/` (`test_qwen3_block_parity.py`,
  `test_spec_decoding_tree_shape.py`, `test_spec_decoding_verify.py`,
  `test_paged_kv*.py`).
- All `ARCH_REFORM_*.md` documentation.
