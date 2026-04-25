# Client placement: CPU vs GPU

BloomBee's client owns three pieces of the stack:

1. **Token embedding** — `embed_tokens.weight[input_ids]`, a lookup.
2. **Remote decoder layers** — sent over the network; the client just
   forwards hidden states to the swarm and receives them back.
3. **Final projection (LMHead)** — `logits = hidden_state @ embed.T`
   when `tie_word_embeddings=True`, or a standalone `lm_head.weight`
   otherwise.

Step 2 is (by design) tiny on the client. Steps 1 and 3 are the only
real local compute and they both scale with `vocab_size × hidden_size`.

## Why this matters for decode throughput

Per decode step the client reads the full embedding / LMHead matrix
at least once. Modern instruct models have much larger vocabularies
than first-gen Llama, so the per-step cost on CPU has crept from
"ignorable" into "dominates wall-clock":

| family          | vocab    | hidden | LMHead numel | relative |
|-----------------|---------:|-------:|-------------:|---------:|
| Llama-7B        |   32 k   |  4096  |   131 M      |    1×   |
| Mixtral-8x7B    |   32 k   |  4096  |   131 M      |    1×   |
| Falcon-40B      |   65 k   |  8192  |   536 M      |    4.1× |
| Qwen3-14B       |  152 k   |  5120  |   778 M      |    5.9× |
| Gemma-4-31B-it  |  262 k   |  5376  |   1.41 B     |   10.7× |

A fp16 matmul of 1.4 B elements on an Ice-Lake-class CPU *without*
AVX-512 takes ~1.2 s. That's your entire decode step just for the
LMHead, before we've even talked to the swarm.

BloomBee's LMHead already logs the slow path:

```
[WARN] [bloombee.client.lm_head.chunked_forward:71] Running the model
in bfloat16 on CPU will be slow since your CPU does not support AVX512.
To speed it up, load the model in float32 using
.from_pretrained(..., torch_dtype=torch.float32)
```

…but `float32` doesn't actually help on a 262k-vocab model — it just
doubles the bytes moved over the DRAM bus.

## Rule of thumb

- **vocab ≤ 64 k** (Llama/Mixtral/Falcon family): CPU client is fine.
  LMHead takes tens of ms, dwarfed by swarm round-trip.
- **vocab 150-200 k** (Qwen3, modern Mistral variants): prefer GPU
  client. CPU client costs 0.3-0.5 s/step and dominates decode.
- **vocab ≥ 256 k** (Gemma-3 / Gemma-4): **GPU client strongly
  recommended**. CPU client costs 1-1.5 s/step and turns decode from
  milliseconds into seconds-per-token.

## How to tell the client is your bottleneck

Run the benchmark with `--cpu_client` and with GPU client, same seed.
If per-step latency shrinks by >50% on GPU, your CPU client is the
bottleneck. Another quick check: watch the server's
`[PIPELINE_EXPOSED_VIEW]` log — the sum of swarm stages should be
much smaller than the client's observed per-step latency.

## What BloomBee does about it

Nothing automatic. Client placement is a user decision because
"light client" is part of the design goal (you might genuinely want to
run on a laptop and accept slow decode for not-owning-a-GPU). Two
escape hatches when you do have a GPU:

```python
# Option A: put the whole model on GPU.
model = AutoDistributedModelForCausalLM.from_pretrained(
    model_id, initial_peers=[...], torch_dtype=torch.float16,
)
model = model.to("cuda")

# Option B: benchmarks/benchmark_inference.py — drop `--cpu_client`.
```

A future BloomBee release may support server-side LMHead ("last layer
as remote block") to remove the client's LMHead cost entirely. Not
there yet.

## References

- Model vocabularies verified from HuggingFace `config.json` on
  2026-04-24: `google/gemma-4-31B-it` (262 144), `Qwen/Qwen3-14B`
  (151 936), `huggyllama/llama-7b` (32 000).
- BloomBee LMHead implementation:
  [`src/bloombee/client/lm_head.py`](src/bloombee/client/lm_head.py).
- Gemma-4 adapter + benchmark results:
  [`.bloombee_local/GEMMA4_DESIGN.md`](.bloombee_local/GEMMA4_DESIGN.md)
  (local, not checked in).
