# Fast LoRA fine-tuning on BloomBee

`DistributedLoRA` keeps the adapters and the optimizer on the client and runs the frozen
backbone on workers. This page lists what makes that fast, what is exact, what is opt-in
and lossy, and the measurements. Reproduce with `benchmarks/benchmark_lora.py`.

## What it does

| Change | Effect | Exact? |
|---|---|---|
| Workers keep each layer's input and autograd graph between forward and backward (`rpc_lora_stream` protocol v2; budgeted, TTL, stateless fallback on any miss) | backward no longer re-uploads inputs/adapters or recomputes the span: ~5L to ~3L layer-passes | yes |
| One Runtime task per span instead of per layer | removes per-layer queue round trips | yes |
| Training-only block for LLaMA/Qwen3: fused SDPA attention, no KV writes, no Task rebuilds, q/k/v and gate/up adapters in one GEMM | faster, much less activation memory | same math as HF SDPA (fp16 rounding differs from eager) |
| Adapters cached on the worker per (client, span, version) | A/B upload only after an optimizer step changed them | yes |
| `DistributedLoRA.train_step(batch, micro_batches=M)` | microbatches pipeline across spans (span k does m while span k+1 does m-1) and spread over replicas; workers sum adapter gradients and the client fetches them once | gradients equal whole-batch gradients (fp32 addition order) |
| `adapter_wire="bfloat16"` (opt-in) | adapters up and gradients down as bf16; fp32 masters/optimizer/sums | lossy, gate-checked |
| `boundary_compression="int8" / "hadamard-int8"` (opt-in) | activations and their gradients as int8 with a bf16 scale per 32 channels | lossy, gate-checked |

Everything lossy is off by default.

## Measurements

Hardware: 2x RTX A6000 (context3), one worker per GPU, client on GPU 0, fp16 base, LoRA
r=16 on all 7 projections, batch 4 x 512 tokens, AdamW. "Eval gap" is the relative
difference of held-out loss between the BloomBee run and HF+PEFT training the same adapter
on the same batches on one GPU (gate: 0.5%).

LAN (local GPUs):

| Model | Run | s/step | tokens/s | speedup | eval gap |
|---|---|---|---|---|---|
| TinyLlama-1.1B | upstream baseline | 2.85 | 719 | 1.00x | 9.2e-4 |
| | cached protocol + fast path | 1.17 | 1745 | 2.43x | 2e-5 |
| | + 4 microbatches | 0.98 | 2095 | 2.91x | 4e-5 |
| | + bf16 adapters | 0.97 | | 2.9x | 3e-5 |
| | + int8 / hadamard-int8 activations | 1.00 / 1.02 | | 2.9x | 2.6e-4 / 2e-5 |
| Qwen3-8B | upstream baseline | 7.23 | 283 | 1.00x | n/a |
| | cached protocol + fast path | 3.20 | 639 | 2.26x | n/a |
| | + 4 microbatches | 2.25 | 910 | 3.21x | 1.9e-4 |

Simulated internet (client link 50 ms latency, 200 Mbit/s, shared by all messages), TinyLlama:

| Run | s/step | speedup | bytes sent / step | eval gap |
|---|---|---|---|---|
| upstream baseline | 12.96 | 1.00x | 151 MB | n/a |
| cached protocol + fast path | 8.53 | 1.52x | 84 MB | n/a |
| + 4 microbatches | 6.77 | 1.91x | 84 MB | n/a |
| + bf16 adapters | 4.75 | 2.73x | 59 MB | 1e-5 |
| + bf16 adapters + hadamard-int8 | 4.20 | 3.09x | 43 MB | 2.3e-4 |

LAN gains come from compute (cached state, SDPA) and pipelining; on slow links the adapter
traffic (about 70 MB each way per step here) dominates, which is what `adapter_wire`
addresses. A worker-side optimizer would remove it entirely but gives up the
client-owned-optimizer design (fault tolerance, checkpointing); not attempted.

## Data parallelism over replicas

When several workers serve the same span, `train_step` spreads microbatches over them. Adapters
and the optimizer stay on the client, so there is no all-reduce: gradients just add up. A
microbatch picks its replicas when it starts (least busy, so faster replicas take more work), and
with replicated spans the number in flight is capped at twice the number of workers.

`DistributedLoRA(..., replica_sync="peer")` (default) keeps the client's traffic independent of
the replica count: the client uploads each adapter version to one replica per span, the others
pull it from there (worker to worker; the unguessable adapter version authorises the pull), and
one leader replica sums the replicas' gradient sums before the client downloads them. A failed
pull falls back to a direct upload. `replica_sync="client"` exchanges adapters and gradients
with every replica (cheaper for 2 replicas on a fast client link).

Worker cache fix found while measuring: kept autograd graphs could fill the whole activation
budget, so later microbatches could not even store their inputs and the client redid the span
statelessly (re-sending activations and a full adapter upload). Old graphs are now dropped
(recomputed in backward) before any "budget" miss is reported. TinyLlama, batch 32 on one
GPU: 6.70 -> 3.81 s/step, 835 MB -> 185 MB sent per step.

Replica scaling, TinyLlama, batch 32 (16 microbatches), s/step. Replica counts above one are
emulated on one GPU with `BLOOMBEE_SIM_COMPUTE` (workers skip the math and sleep the time
measured on a dedicated A6000: 2.19 us forward, 5.0 us backward per layer-token, 13 ms per task;
emulated 1 replica 3.90 s vs real 3.81 s). Emulation checks timing and the protocol, not
accuracy; accuracy of the replicated path is covered by `tests/run_lora_replicas.py`.

| Replicas | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| light adapters (q, v; r=8), peer sync | 3.38 | 2.01 | 1.46 | 1.62 |
| heavy adapters (7 projections; r=16, 50 MB), peer sync | 3.93 | 2.86 | 2.95 | 4.29 |
| heavy adapters, client sync | 3.93 | 2.49 | 7.04 | n/a |

Simulated 50 ms / 200 Mbit/s client link, heavy adapters: 1 replica 11.0 s, 2: 11.7 s (client
sync 15.1 s), 4: 11.6 s (client sync 27.7 s). Client bytes per step with 4 replicas: 185 MB
(peer sync, same as one replica) vs 337 MB (client sync).

What limits it: every activation still crosses the client link, so the speedup saturates near
2.3x at 4 replicas and replicas cannot help when that link is the bottleneck (use
`boundary_compression` / `adapter_wire` there). Heavy adapters add a fixed per-step sync (upload,
pull, reduce, download) that data parallelism cannot shrink; the leader sums its peers' gradients one
at a time (fetching them all concurrently was 3x slower with 7 peers: the leader's handler process,
not the link, is the limit), so 8 replicas with heavy adapters get slower than 4. A tree reduce
is the next step; with light adapters 8 replicas are still within 11% of 4.

## Usage

```bash
# every worker
BLOOMBEE_ENABLE_LORA=1 python -m bloombee.cli.run_server <model> ...

# client
python examples/train_lora.py --model <model> --initial_peers ... --data train.jsonl \
    --output out --micro_batches 4                    # exact
#   add --adapter_wire bfloat16 --boundary_compression hadamard-int8 on slow links
```

```python
model = DistributedLoRA(base, LoRAConfig(...), adapter_wire="bfloat16")  # options are keyword-only
loss = model.train_step(ids, labels=ids, micro_batches=4)                  # backward included
optimizer.step()
```

Benchmark and accuracy gate:

```bash
python benchmarks/benchmark_lora.py --model Qwen/Qwen3-8B --gpus 0,1 --batch 4 --seq 512 \
    --use-train-step --micro-batches 4 --reference --sim-link 50,200 --output out/run
```

Worker switches are in `README.environment-switches.md`.

## Notes

- Cached backward state is per forward. If a worker restarts or the route changes between
  forward and backward, the client transparently re-sends (stateless path, same gradients).
- With `accumulate_on_workers` (default in `train_step`) a worker that loses its gradient
  sums before `collect` makes the step fail loudly ("redo the step") instead of silently
  dropping gradients.
- Models: LLaMA and Qwen3 dense blocks (as before). Tensor-parallel blocks are unsupported.
