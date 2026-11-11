import time
import torch
from transformers import AutoTokenizer
from bloombee import AutoDistributedModelForCausalLM

INITIAL_PEERS = ["/ip4/10.140.81.87/tcp/31340/p2p/QmQYXamg7hH8RabuJr4Wk3MoeanKL5qWsV4UgkDX235Bj3"]
MODEL_NAME = "unsloth/DeepSeek-V3-bf16"
PROMPT = "What is the capital of China?"  # <-- change this to whatever you want to ask

print(f"Loading tokenizer for {MODEL_NAME} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
messages = [{"role": "user", "content": PROMPT}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)

print("Connecting to swarm and loading distributed model ...", flush=True)
model = AutoDistributedModelForCausalLM.from_pretrained(
    MODEL_NAME,
    initial_peers=INITIAL_PEERS,
    torch_dtype=torch.bfloat16,
    # Worker-to-worker relaying (the default) was previously found to hang on a broken
    # leg of the relay chain between two specific workers -- invisible to client-side
    # connectivity checks, since the client isn't on that path. Route every hop through
    # the client instead, same escape hatch already used by test_prompt_inference.py /
    # benchmarks/benchmark_inference.py's --use_server_to_server flag.
    use_server_to_server=False,
)

print(f"Prompt: {PROMPT!r}", flush=True)
start = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=24, do_sample=False)
elapsed = time.perf_counter() - start

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Response: {response!r}", flush=True)
print(f"Elapsed: {elapsed:.2f}s", flush=True)
