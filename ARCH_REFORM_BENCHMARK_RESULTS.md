# arch-reform Benchmark Results

Measured on V100-SXM2-16GB (sm_70, torch 2.5.1+cu121), CPU side on
2.5GHz Intel (V100 host). Numbers are median over 20 iterations after
3 warmup calls. Reproduce with:

```
BLOOMBEE_BENCH_CUDA=1 python tests/bench_arch_reform.py
```

## 1. Ancestor matrix construction — `spe_dec_tree.build_ancestor_matrix_optimized`

Per-step cost during speculative verify. Called once per decode step on
the drafter tree.

| Tree | n | old (matmul closure) | new (parent walk) | speedup |
|---|---:|---:|---:|---:|
| depth=3 width=2 (BloomBee default) | 7 | 0.271 ms | 0.061 ms | **4.5×** |
| depth=4 width=2 balanced | 15 | 0.345 ms | 0.071 ms | **4.8×** |
| width=8 root-only | 9 | 0.220 ms | 0.060 ms | **3.7×** |
| linear chain | 32 | 2.062 ms | 0.193 ms | **10.7×** |
| unbalanced | 30 | 0.451 ms | 0.096 ms | **4.7×** |

Complexity went from O(n³) with early-exit to O(n·depth). Larger/deeper
trees amortize more — enables cheap experimentation with larger trees.

## 2. Spec-dec verify argmax hoist — `speculative_model._extract_best_verified_paths_fixed`

Per-step verification cost. The win is on CUDA: moving per-node
`.item()` calls (each of which forces a CUDA sync) to a single batched
`argmax → tolist`.

| Device | batch | old (per-node .item) | new (batched → tolist) | speedup |
|---|---:|---:|---:|---:|
| CPU | 1 | 0.795 ms | 0.594 ms | 1.3× |
| CPU | 4 | 3.182 ms | 1.906 ms | 1.7× |
| CPU | 16 | 12.745 ms | 7.510 ms | 1.7× |
| **CUDA** | 1 | 0.392 ms | 0.146 ms | **2.7×** |
| **CUDA** | 4 | 1.356 ms | 0.150 ms | **9.0×** |
| **CUDA** | 16 | 5.263 ms | 0.166 ms | **31.7×** |

At batch=16 the old path spent 5.1 ms just syncing argmax to host per
step; the new path holds that flat at ~0.17 ms. Scales with batch and
tree size.

## 3. FlexGen in-place KV write — `pytorch_backend.mha_gen_llama` slab path

Per-step decode cache write. Small wins on CPU, break-even on CUDA in
isolation.

| Device | batch × history | old (`torch.cat`) | new (in-place slab) | speedup |
|---|---|---:|---:|---:|
| CPU | 1 × 63 | 0.029 ms | 0.012 ms | 2.3× |
| CPU | 1 × 1023 | 0.113 ms | 0.012 ms | 9.1× |
| CPU | 4 × 511 | 1.287 ms | 0.014 ms | **93.2×** |
| CUDA | 1 × 63 | 0.070 ms | 0.081 ms | 0.9× |
| CUDA | 1 × 1023 | 0.075 ms | 0.078 ms | 1.0× |
| CUDA | 4 × 511 | 0.111 ms | 0.077 ms | 1.4× |

**Read the CUDA numbers carefully.** At B=1 the isolated per-call cost
is identical — both paths are dominated by kernel launch, not memcpy.
The actual production win is not in this microbench:

- **Allocation churn eliminated.** `torch.cat` allocates a new tensor
  every decode step. Over 512 decode steps on a 16GB GPU this produces
  a growing-sized alloc per step and creates fragmentation; eventually
  peak memory forces a realloc. The slab allocates once.
- **Unblocks paged KV.** The slab is the substrate for Phase 2 — the
  PagedKVTable primitive relies on write-in-place semantics; the old
  `torch.cat` path cannot be paged.

The B=4 × 511 case (16 MB history) is the first point where memcpy cost
exceeds kernel launch and the win shows even in isolation (1.4×).

## 4. End-to-end decode simulation — combined effect

Combines the three optimizations over a full speculative decode loop —
each step runs: ancestor-matrix rebuild → tree verify → one accepted KV
write. Llama-7b geometry (n_head=32, head_dim=128, vocab=32000, tree
depth=3 width=2). Reproduce with:

```
BLOOMBEE_SIM_CUDA=1 python tests/sim_end_to_end_decode.py
```

### Short generation (prompt=512, n_steps=128)

| Batch | old total | old ms/step | new total | new ms/step | speedup | peak mem reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 255.2 ms |  1.99 ms | 188.1 ms | 1.47 ms | **1.36×** |  −4.1 MB |
| 4  | 378.3 ms |  2.96 ms | 188.6 ms | 1.47 ms | **2.01×** | −15.3 MB |
| 16 | 945.7 ms |  7.39 ms | 193.4 ms | 1.51 ms | **4.89×** | −61.4 MB |

### Long generation (prompt=1024, n_steps=256)

| Batch | old total | old ms/step | new total | new ms/step | speedup | peak mem reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1  |  497.7 ms |  1.94 ms |  374.4 ms | 1.46 ms | **1.33×** |   −8.8 MB |
| 4  |  780.8 ms |  3.05 ms |  369.6 ms | 1.44 ms | **2.11×** |  −35.2 MB |
| 16 | 2071.5 ms |  8.09 ms |  384.2 ms | 1.50 ms | **5.39×** | −141.4 MB |

**Key observation: the new path holds ~1.5 ms/step flat across all batch
sizes and all generation lengths.** The old path scales roughly linearly
with both batch (dominated by per-node `.item()` sync) and step count
(dominated by `torch.cat` allocation churn). At batch=16 with a 256-step
decode, the arch-reform path is **5.4× faster** and uses **141 MB less
peak GPU memory**.

The measured per-step time (1.5 ms) tracks the sum of the 3 isolated
benchmarks above:
- ancestor matrix: ~0.06 ms
- verify hoist (B=16): ~0.17 ms
- in-place KV write: ~0.08 ms
- plus fixed per-step overhead (randn, tensor allocation for logits)

So the speedup isn't mystical — it's exactly the 3 isolated wins
compounding, with no regressions in the code paths we didn't touch.

## Summary

- Phase 3 ancestor-matrix: 4–11× per step, across all realistic SD tree
  shapes.
- Phase 3 verify hoist: up to **32×** on CUDA at batch=16 — this is the
  dominant per-step CPU↔GPU sync in BloomBee today.
- Phase 0 in-place slab: break-even per call on CUDA, but the real win is
  structural (no alloc churn, enables paged KV).
- **Combined end-to-end: 5.4× at batch=16 over a 256-step decode, with
  141 MB less peak GPU memory.** See section 4.

## Not yet measured

Phase 2 MemoryCache↔PagedKVTable shim is landed behind
`BLOOMBEE_PAGED_KV=1`. 13 unit tests (substrate byte-equivalence +
per-handle decoupling + rollback hiding) pass on V100. End-to-end
throughput impact via the server still requires unblocking the
`flex_llama.py:441` hardcoded `huggyllama/llama-7b` snapshot_download
that triggers on any model, plus FlexGen's numpy-layout weight
conversion pipeline that safetensors-native TinyLlama doesn't satisfy.
