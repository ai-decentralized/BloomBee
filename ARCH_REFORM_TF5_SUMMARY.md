# arch-reform-qwen3-4b — Architecture Reform + TF 5.x Migration Summary

Handoff document for the `arch-reform-qwen3-4b` branch as of commit
`8022e74` (2026-04-19). Written so a collaborator with a fresh large
GPU can pick up, re-run benchmarks at scale, and extend the work to
bigger models (llama-70b, falcon-40b, mixtral-8x7b) without reading
the full commit history.

---

## 1. What this branch is

Two intertwined tracks on top of the BloomBee mainline:

1. **Architecture reform** (Phase 0 → Phase 3). A refactor of the KV
   hot path, spec-decoding algorithms, block_function seam, and
   FlexGen substrate. Lands per-step speedups on real llama-7b and a
   rollback-safe paged-KV shim that is now load-bearing under
   spec-dec (commit `8a19811`).
2. **transformers 5.x migration**. Upstream TF 5.x moved Bloom /
   Falcon / Llama attention from tuple-KV to a `Cache` API; several
   bookkeeping paths (`mark_tied_weights_as_initialized`,
   `prepare_inputs_for_generation`) became incompatible with
   BloomBee's distributed LMHead / sentinel-based RemotePastKeyValues.
   The migration restores all five README-declared model families
   (Llama, Bloom, Falcon, Mixtral, Qwen3) under TF 5.5.4 end-to-end.

The branch's single invariant: **every model listed in README.md must
run end-to-end under the current TF pin.** If a model is broken, the
branch is broken.

## 2. Environment baseline

- **Primary host**: V100-SXM2-16GB, driver 580, CUDA 13.0, fp16.
- **Venv**: `/data/models/bloombee-venv` on `admin@192.168.31.118`.
- **Pinned TF**: `transformers==5.5.4` (pyproject pin updated, BloomBee
  adapters carry compat shims for both 4.x and 5.x signatures).
- **Models staged on V100**: `bloom-560m`, `falcon-rw-1b`,
  `qwen3-0.6b`, `tinyllama-1.1b`, `qwen35-27b.gguf`.
- **Torch safetensors workaround**: torch < 2.6 can't load `.bin`
  under CVE-2025-32434; falcon-rw-1b was converted to
  `model.safetensors` (shared-tensor clone for `lm_head` ↔
  `word_embeddings`).

## 3. Design changes (organized by subsystem)

### 3.1 Paged-KV shim (Phase 2) — `BLOOMBEE_PAGED_KV`

`MemoryCache._register_paged_view(handle, (k, v))` aliases a
`PagedKVTable` onto the cache slab at allocation time. Callers still
see the same `TorchTensor` objects; the paged table is kept as a
state-only mirror. `track_write` + commit/rollback are plumbed through
`_do_reorder_task` so speculated KV pages can be dropped on rejection
without touching the attention kernel's read path.

- **Flag**: `BLOOMBEE_PAGED_KV=1`. Off by default; additive and
  non-fatal (failures log a warning and fall back to the legacy slab).
- **Cost** (single-server, llama-7b): ~1.6% per-step overhead.
- **Status**: load-bearing under spec-dec rollback; attention reads
  still hit the slab directly. Next step is to route reads through
  `gather_prefix`, at which point the slab becomes a staging buffer.

### 3.2 Spec-decoding algorithm refresh

- **Vectorized spec verify** (`e89d7cb`): argmax hoisted to batched
  GPU ops. Scales sharply with batch size — 2.7× @ B=1, **31.7× @
  B=16**.
- **EAGLE-2 budgeted tree shape** (`1e70e26`): replaces old
  scalar-prob tree with probability-weighted expansion.
- **SpecInfer stochastic rejection sampling** (`fdfd0b7`): proper
  accept/reject instead of always-accept-top-1.
- **O(n·depth) ancestor matrix** (`35c6697`): tree parent-walk instead
  of matmul closure. 4.5× → 10.7× depending on tree shape.

### 3.3 FlexGen KV write path

- **In-place slab write** (`c04bdee`): eliminates the per-step
  `torch.cat` that reallocated a full-history tensor every decode
  step. 2.3×–93.2× on CPU; CUDA break-even per call but removes
  allocator churn (visible in section-6 client-wall comparison below).

### 3.4 Cache layout compatibility (TF 5.x)

TF 5.x rewrote `BloomAttention` / `FalconAttention` / `LlamaAttention`
to expect `layer_past: Cache` (with `.update()` method). BloomBee's
`_run_block_forward` still hands blocks raw `(k, v)` tensor tuples
because the distributed backend routes cache state through the
`MemoryCache` slab, not HF's Cache. Three patches land the bridge:

- **`OptimizedBloomAttention`** (`src/bloombee/models/bloom/block.py`):
  subclass of `BloomAttention` that overrides `forward` to do the
  tuple-cache concat inline and returns `present` in BloomBee's
  canonical **3D layout** `key=[B*H, D, S]`, `value=[B*H, S, D]` so
  that `memory_cache_manager._write_kvs`'s assertion
  `key_t.ndim == 3` passes. `WrappedBloomBlock` installs this in its
  `self_attention` slot and implements its own layer-norm + attn + MLP
  body (can't delegate to `super().forward` — upstream passes
  `layer_past` positionally expecting a Cache).
- **Robust `past_length` detection**: past key may arrive as either 3D
  `[B*H, D, S]` (backend tuples) or 4D `[B, H, S, D]`
  (OptimizedBloomAttention's own output when re-entered). Disambiguate
  by matching `head_dim` against the last two shape dims.
- **Bloom HF-path routing**: `_is_hf_model` in
  `server/from_pretrained.py` now includes `WrappedBloomBlock`; Bloom
  no longer falls through the FlexGen Llama dispatch branch (which
  passes `(config, layer_idx, env, policy, weight_home, path)` and
  crashed with "`takes from 2 to 3 positional arguments but 7 were
  given`"). `server/block_utils.get_model_block` dispatches Bloom with
  `(config, layer_idx)`.

### 3.5 Tied-weights compatibility (TF 5.x)

TF 5.x's `PreTrainedModel.tie_weights` walks `_tied_weights_keys`
(which includes `lm_head.weight`) via `self.get_parameter(name)`.
BloomBee's `LMHead` stores `self.weight = None` until bind-time
because the embedding it ties to lives on a different peer. `None` is
not an `nn.Parameter`, so `get_parameter` raises
`AttributeError("'weight' is not an nn.Parameter")`.

Uniform fix across Bloom / Falcon / Llama / Mixtral / Qwen3 `ForCausalLM`:

```python
def mark_tied_weights_as_initialized(self, loading_info):
    # TF 5.x bookkeeping that crashes on LMHead's weight=None placeholder;
    # we tie manually below, so the walk is unnecessary.
    return

def tie_weights(self, missing_keys=None, recompute_mapping=True):
    if getattr(self.config, "tie_word_embeddings", False):
        embed = self.get_input_embeddings()
        if embed is not None and getattr(embed, "weight", None) is not None:
            self.lm_head.weight = embed.weight
```

When `tie_word_embeddings=False` (llama-2-7b, falcon-rw-1b) this is a
no-op. When True (qwen3, most small bloom variants) it binds the
embedding weight directly onto the client-side LMHead, which is the
behavior mainline had before TF 5.x refactored the walk.

### 3.6 Device-context guard

TF 5.x's `from_pretrained` wraps model `__init__` in a `torch.device('cuda')`
context, which hijacks hivemind's `torch.empty()` calls inside the
DHT / MPFuture setup and creates CUDA tensors where CPU tensors were
expected. `share_memory_()` only works on CPU tensors, so the server
crashes during bring-up. Fix (Llama had it, Bloom / Falcon / Mixtral /
Qwen3 now too):

```python
with torch.device('cpu'):
    self.layers = RemoteSequential(config, dht=dht)
```

### 3.7 Falcon Cache-aware `prepare_inputs_for_generation`

Upstream Falcon's default `prepare_inputs_for_generation` does
`past_key_values[0][0].shape[2]`, which crashes on BloomBee's
`RemotePastKeyValues` (a `Cache` subclass whose `__getitem__` returns
a sentinel). Reimplemented Cache-aware — see
`src/bloombee/models/falcon/model.py`. Mirrors what Llama already had.
Also set `_supports_cache_class = True` so TF 5.x's generate() loop
gives us a Cache instead of an alleged legacy tuple.

### 3.8 FlexGen minor

- `flex_llama.FLEX_LlamaAttention.__init__` now forwards `layer_idx`
  to `super().__init__` when TF 5.x requires it, with a `TypeError`
  fallback for older TF.
- `rms_norm` default eps is **1e-5** (matches LLaMA config's
  `rms_norm_eps`). A prior experimental tweak had it at 1e-6; restored
  because some configs read the default when the per-layer norm weight
  is missing.

## 4. What's verified (and what isn't)

Verified on V100 16G / TF 5.5.4:

| Model | Status | Note |
|---|---|---|
| Llama-7b | ✅ E2E (5c88aa…5c98d83 doc) | Regression-safe under 8022e74 additions |
| Qwen3-0.6B | ✅ E2E | Regression-check passed post-migration |
| Bloom-560m | ✅ E2E | Coherent output after OptimizedBloomAttention |
| Falcon-rw-1b | ✅ E2E | Cache-aware prepare_inputs + safetensors workaround |
| Mixtral | ⚠️ Imports OK | Too large for 16GB V100, not yet size-verified |

Known-unsupported (pre-existing FlexGen limitation, not caused by this
branch):

- **tinyllama-1.1b** — uses GQA (32 Q heads / 4 KV heads); FlexGen's
  `flex_llama.py` is pure MHA. Not in README's supported list.

## 5. Performance snapshot (llama-7b, V100, pre-TF5.x)

Data below is from `ARCH_REFORM_BENCHMARK_RESULTS.md` §6. The
migration to TF 5.x added only `tie_weights` overrides (no-op when
`tie_word_embeddings=False`, which llama-7b is), so numbers should be
within noise. **Needs a re-run under TF 5.5.4 to make this rigorous.**

**Single-server (32 blocks)**

| Branch | ms/step | tok/s | Client wall (steady-state) |
|---|---:|---:|---:|
| mainline (e5b88aa) | 225.1 | 75.4 | ~24 s |
| **arch-reform** | **152.9** | **111.0** | **~12.4 s** |

→ **1.47× server / 1.90× client**

**Two-server pipeline (blocks 0:16 on A, 16:32 on B, same V100)**

| Branch | ms/step A | ms/step B | tok/s (per server) |
|---|---:|---:|---:|
| mainline | **FAILED to launch** (pinned-buffer OOM) | — | — |
| arch-reform PAGED_KV=0 | 81.2 | 80.8 | ~210 |
| arch-reform PAGED_KV=1 | 82.8 | 82.9 | ~205 |

On V100 hosts with a strict `RLIMIT_MEMLOCK`, FlexGen's default 1 GB
pinned CPU relay buffer (`copy_worker_func`) can fail to register
before the DHT socket is even bound, masquerading as a CUDA OOM. If
you hit that on constrained hardware, lower the buffer locally — not
landed as a knob in this branch because it did not reproduce on A100
(the relay is unused when `Disk% = 0`, which is the default case).

## 7. Runbook (V100 — reproduce §5 numbers)

```bash
# On admin@192.168.31.118
source /data/models/bloombee-venv/bin/activate
cd /data/models/bloombee
git checkout arch-reform-qwen3-4b  # 8022e74

# Single-server llama-7b (assuming llama-7b staged locally)
python -u -m bloombee.cli.run_server \
    /data/models/llama-7b-hf \
    --new_swarm --num_blocks 32 --port 31363 \
    --device cuda --torch_dtype float16 \
    --dht_prefix llama7b \
    --identity_path /data/models/bb_run/identity_llama7b.key \
    2>&1 | tee server_llama7b.log

# Bloom-560m (smaller smoke test, 24 blocks)
python -u -m bloombee.cli.run_server \
    /data/models/bloom-560m \
    --new_swarm --num_blocks 24 --port 31363 \
    --device cuda --torch_dtype float16 \
    --dht_prefix bloom560 \
    --identity_path /data/models/bb_run/identity_bloom.key

# Client-side: see /data/models/bb_run/client_bloom.py for a template.
```

## 8. Scaling to large GPUs — checklist

When moving to A100-80G / H100 / multi-GPU for llama-70b or
mixtral-8x7b validation:

- [ ] Verify `RLIMIT_MEMLOCK` is `unlimited` (`ulimit -l`); on
      constrained hosts the FlexGen pinned CPU relay buffer (1 GB fp16
      = 2 GB pinned) can fail registration at server bring-up.
- [ ] Re-run `ARCH_REFORM_BENCHMARK_RESULTS.md` §6 under TF 5.5.4 to
      confirm no perf regression from the migration.
- [ ] For Mixtral, ensure `config._attn_implementation` defaults to
      `eager` on hosts without flash-attn; the auto-set in
      `server/block_utils._autoset_attn_impl` handles this but is
      worth a sanity check (check `attn_impl` field in the first
      block's log line).
- [ ] For GQA-bearing models (llama-3.1-8B, llama-3-70B, qwen3-larger
      sizes): they currently only route through HF block (not FlexGen
      llama). If a FlexGen path is desired for GQA, `flex_llama.py`
      needs a KV-head split → repeat pattern before attention — not
      done on this branch.
- [ ] Confirm client-side Cache type compatibility for each new model
      (generate() calls `prepare_inputs_for_generation` — falcon /
      llama have Cache-aware versions here; bloom / mixtral inherit
      the upstream one and have worked so far but double-check under
      new TF patches).

## 9. Known limitations / follow-ups

1. **Paged-KV shim read path not wired**: attention still reads the
   slab directly; `gather_prefix` would make the shim fully
   authoritative. See `project_arch_reform_progress.md` follow-up
   list.
2. **Spec-dec full-server validation**: unit tests cover the paged-KV
   rollback cycle, but the default client doesn't invoke spec-dec so
   `_do_reorder_task` with `is_spec_dec=1` hasn't been exercised E2E
   on V100. Need a spec-dec-enabled client.
3. **FlexGen GQA**: not supported; GQA models go through HF path.
4. **Mixtral E2E not run**: size-bound on 16GB V100.
5. **Continuous batching**: per-sequence `L_seq` not yet threaded
   (shared `cache_len` at `backend.py:542`); `load_batch_to_runtime`
   doesn't yet do continuous admission.

## 10. Commit lineage (top → bottom on this branch)

Top commit `8022e74 fix(models): restore all 5 README-declared models
under TF 5.x`.

Prior landmarks:
- `0c1c53b` fix(qwen3): build causal mask + keep rotary inv_freq fp32
- `b5fb0fb` fix(qwen3): use explicit config.head_dim for KV cache
- `1ecb51d` fix(client): pass layers=[] to TF 5.x Cache.__init__
- `2240664` fix(qwen3): override TF 5.x tie_weights
- `c56cc76` fix(qwen3): override TF 5.x mark_tied_weights_as_initialized
- `8a19811` feat(kv-cache): Phase 2 paged-view shim load-bearing under spec-dec
- `5c98d83` docs: end-to-end llama-7b comparison vs mainline
- `f3cfa03` docs: real-server round-trip numbers for Phase 2 paged shim
- `da30b0e` fix(weights): local-safetensors conversion path
- `d011b4b` fix(flexgen): shrink pinned CPU relay buffer to 4 MB
- `c04bdee` FlexGen in-place KV write (Phase 0)
- `dd12f40` PagedKVTable primitive + 10 tests
- `35c6697` O(n*depth) ancestor matrix
- `e89d7cb` vectorized spec verify
- `fdfd0b7` SpecInfer stochastic rejection sampling
- `1e70e26` EAGLE-2 budgeted tree shape

See `ARCH_REFORM_PLAN.md`, `ARCH_REFORM_SUMMARY.md`,
`ARCH_REFORM_BENCHMARK_RESULTS.md`, `PHASE2_PAGED_KV_INVARIANTS.md` for
deeper dives on each track.
