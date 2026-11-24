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
