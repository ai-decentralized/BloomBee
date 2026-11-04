# Standard LoRA training

BloomBee supports an opt-in standard LoRA training path for dense LLaMA and Qwen3:

```
y = W x + (lora_alpha / r) B A dropout(x)
```

Base weights remain frozen. A uses Kaiming-uniform initialization and B starts
at zero. Only A/B are trainable. Supported projections: `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; defaults are q/v.

The client owns FP32 A/B parameters and optimizer state. Each worker computes
its span's forward and input/A/B gradients through the real block implementation.
For LLaMA the update is injected at FlexGen's actual functional projections, not
the unused `nn.Linear` parameter shells. Qwen3 uses scoped HF projection hooks.

Each RPC carries immutable adapter tensor copies. Backward recomputes with the
same saved copies and dropout seed. No server optimizer, gradient accumulation
or adapter mutation occurs. Ordinary PyTorch optimizers and accumulation work on
the client, including when input embeddings do not require gradients. Updating
parameters between a forward and its backward is rejected by PyTorch's version
check. Concurrent clients' adapters are isolated by request context.

This correctness-oriented implementation sends A/B at forward and backward,
and relays span gradients through the client. Its bandwidth cost should be
measured before large-scale use. It does not modify the existing cached S2S
backward protocol or promise S2S optimizer synchronization.

## Workers

Use the same checkout on all peers. Enable LoRA **before** starting every worker:

```bash
export BLOOMBEE_ENABLE_LORA=1
python -m bloombee.cli.run_server /path/to/llama-model \
  --initial_peers "$BOOTSTRAP" --dht_prefix my_lora_model \
  --block_indices 0:16 --torch_dtype float16 --batch_size 2
# A second worker serves 16:32 with the same model/prefix/bootstrap.
```

The LoRA stream is disabled by default. Workers continue serving unchanged base
inference alongside stateless LoRA requests. LoRA tasks use the existing Runtime
queue, avoiding concurrent mutation of FlexGen working buffers.

## Python API

```python
import torch
from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig

base = AutoDistributedModelForCausalLM.from_pretrained(
    "/path/to/llama-model", initial_peers=[bootstrap],
    dht_prefix="my_lora_model", torch_dtype=torch.float16,
).cuda()
model = DistributedLoRA(base, LoRAConfig(r=8, lora_alpha=16)).cuda().train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)

optimizer.zero_grad(set_to_none=True)
loss = model(input_ids=ids, labels=ids, use_cache=False).loss
loss.backward()
optimizer.step()
model.save_pretrained("adapter")

# Load with a fresh base model. Also readable by HuggingFace PeftModel.
loaded = DistributedLoRA.from_pretrained(fresh_base, "adapter").cuda().eval()
tokens = loaded.generate(ids, max_new_tokens=16)
```

`save_pretrained` writes PEFT-format `adapter_config.json` and
`adapter_model.safetensors`, containing only A/B. `--resume_adapter` loads these
weights with a fresh optimizer. For a complete training resume, clear gradients
after an optimizer step and save a new checkpoint directory:

```python
optimizer.zero_grad(set_to_none=True)
model.save_training_checkpoint("checkpoint-100", optimizer, step=100)

model = DistributedLoRA.from_pretrained(fresh_base, "checkpoint-100").cuda().train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
state = model.load_training_state("checkpoint-100", optimizer)
# Sample the next batch only after load_training_state restores torch RNG.
```

Checkpoints include A/B, optimizer state, completed optimizer steps, and client
torch CPU/CUDA RNG states. Loading checks the adapter values/configuration,
optimizer type and parameter order. Checkpoints belong at optimizer boundaries;
pending accumulated gradients are rejected. This API does not save a caller's
custom scheduler, scaler, Python/NumPy RNG or dataloader state.

## Text training example

`examples/train_lora.py` reads local JSONL `{"text": "..."}`, appends EOS, and
packs fixed-length unpadded causal chunks. It supports gradient accumulation:

```bash
PYTHONPATH=src python examples/train_lora.py \
  --model /path/to/model --initial_peers "$BOOTSTRAP" \
  --dht_prefix my_lora_model --data train.jsonl --output adapter \
  --steps 100 --seq_len 128 --batch_size 1 --gradient_accumulation_steps 4
```

Add `--checkpoint_every 10` to save periodic `checkpoint-N` directories. A final
checkpoint is always saved. Resume in a fresh output directory, keeping the
model, data and training settings the same:

```bash
PYTHONPATH=src python examples/train_lora.py \
  --model /path/to/model --initial_peers "$BOOTSTRAP" \
  --dht_prefix my_lora_model --data train.jsonl --output continued \
  --resume_checkpoint adapter/checkpoint-100 \
  --steps 200 --seq_len 128 --batch_size 1 --gradient_accumulation_steps 4
```

`--steps` is the total target optimizer step count. The example checks a hash of
the packed training tokens and the batch/accumulation/device/lr settings before
resuming. The same frozen base weights and deterministic worker implementation
are required for exact continuation; a changed model under the same path is not
detected by the token hash.

## Deploy a fixed adapter through ordinary KV inference

Export the trained adapter into a new HF model directory:

```bash
python examples/merge_lora.py --model /path/to/base --adapter adapter --output /path/to/merged
```

The exporter loads the base on CPU in FP32, merges with PEFT, and saves FP16
weights plus the tokenizer. It needs RAM for the FP32 base. Output must not
already exist. Start ordinary BloomBee workers and a fresh ordinary client using
the merged directory and a **new DHT prefix**. These weights can use the existing
KV session and inference pipeline. The LoRA RPC need not be enabled.

This is fixed-weight deployment: merging FP16 weights can introduce rounding
differences from separate A/B computation. It does not implement dynamic adapter
switching inside a KV session or worker-resident adapter updates.

## Current limits

- Rank 1..256; bias-free alpha/r LoRA. DoRA, rsLoRA, per-layer rank/alpha,
  trainable base biases and modules_to_save are not implemented.
- Unpadded causal sequences only. `labels=-100` may mask loss positions, but
  arbitrary attention masks, custom positions, prompt tuning and speculative
  decoding are not supported by this training protocol.
- Single-device blocks; no tensor parallelism. LoRA does not use inference KV
  caches. `DistributedLoRA.generate()` is greedy full-prefix recomputation,
  independent of base-model KV sessions. Calling the wrapped base model's
  regular cached `generate()` is rejected, so adapters cannot be silently bypassed.
- Base weights are frozen in eval mode at the worker; LoRA dropout is separately
  controlled by the client train/eval state. Standard LLaMA/Qwen3 attention
  dropout defaults to zero. No inference microbatch or KV-offload feature is
  automatically treated as training activation scheduling.
- LLaMA FlexGen and Qwen3 HF weight offload are tested at 50% GPU placement.
  Qwen3 backward retains staged GPU storage for one recomputed block until its
  gradients finish, then releases it; CPU parameter homes remain unchanged.
  This does not imply that the entire training working set follows the inference
  weight-offload ratio.
- FP16 workers/client are recommended for the LLaMA FlexGen path. Requests cast
  to the actual projection dtype and return the client's input dtype. FP32 adapter master weights
  and their gradients retain their exact FP32 values when the surrounding model
  is cast to half/bfloat16/double. Weight offload and other
  configurations require the corresponding integration tests; tests are not a
  claim of compatibility with every model/offload combination.

## Validation

`tests/test_standard_lora.py` compares every supported linear projection with
independent PEFT forward/dX/dA/dB, tests initialization, accumulation, frozen
base parameters and deterministic dropout recompute.

`tests/run_lora_rpc.py` launches a private DHT and one or two **real worker
processes on CUDA**. It uses complete two-layer random LLaMA/Qwen3 causal models
with GQA, compares logits/loss/LoRA gradients/SGD updates with HF PEFT, checks
full-batch vs accumulated microbatches, runs AdamW training, saves/reloads via
both BloomBee and PEFT, and confirms ordinary base output is unchanged:

```bash
PYTHONPATH=src python tests/run_lora_rpc.py --family llama --stages 2 --output /tmp/lora-check
```

`--check-resume` additionally runs the actual training CLI in separate processes
and compares continuous four-step training with two steps followed by a resume.
It enables 0.3 LoRA dropout and two-step gradient accumulation, and requires
exact adapter/optimizer/RNG/loss-history equality. `--check-merged` also runs the
export CLI and checks ordinary remote KV prefill plus three decode steps against
the merged HF model, with two-sample batches split into size-one microbatches
and the lossless wrapper enabled. It also compares native greedy `generate()`
tokens with HF and, with two workers, requires actual S2S/microbatch log samples.
Use `--cache-gpu-percent 50 --weight-gpu-percent 50` to test LLaMA's combined
weight and KV offload configuration.

These tests validate distributed training correctness on tiny models, not
pretrained-model task quality, speedup or physical cross-host network behavior.

`tests/run_lora_pretrained.py --model /existing/local/llama-7b --output /tmp/lora-7b`
runs a short two-worker training/save/reload smoke test against an existing
checkpoint. It uses no downloads; its short fixed sample is not a task-quality
evaluation.

Validated on 2026-11-04 with Python 3.10, PyTorch 2.5.1+cu124,
Transformers 5.16.1, PEFT 0.8.2 and one Quadro RTX 6000. Both workers ran
as separate processes on that host:

| Check | Result |
| --- | --- |
| Complete CPU CI suite | 275 passed, 12 CUDA tests skipped |
| LoRA, task metadata and cache regressions with CUDA | 50 passed |
| Two-worker tiny LLaMA, all seven targets, 25 AdamW steps | Loss 4.099141 to 3.596226; maximum PEFT gradient error 1.83e-5 |
| Two-worker tiny Qwen3, all seven targets, 25 AdamW steps | Loss 4.084572 to 3.591455; maximum PEFT gradient error 2.28e-5 |
| CLI resume with dropout and accumulation, both families | Adapter, optimizer, RNG and loss history exactly equal to uninterrupted training |
| Merged KV inference, both families | Prefill and three decode steps match HF within FP16 tolerance; greedy tokens exactly equal |
| Pretrained LLaMA 7B, 32 layers split 16/16, q/v rank 8, three AdamW steps | Loss 2.022018 to 1.236878; nonzero gradients; adapter reload exactly equal |

The tiny-model runs used 50% GPU weight placement; LLaMA also used 50% GPU KV
placement for merged inference. Both merged runs used batch size two,
microbatch size one, the lossless wrapper and actual S2S calls. Reproduce them
with `--stages 2 --weight-gpu-percent 50 --check-resume --check-merged`, adding
`--cache-gpu-percent 50` for LLaMA. These results establish the tested numerical
and lifecycle behavior, not task quality or cross-host performance.

Standard LoRA definition and initialization follow
[Microsoft LoRA](https://github.com/microsoft/LoRA) and
[HuggingFace PEFT](https://huggingface.co/docs/peft/main/en/package_reference/lora).
