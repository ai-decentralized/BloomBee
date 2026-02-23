<p align="center">
    <img src="figures/bloombee.jpg" alt="Bloombee Logo" /><br>
    Run large language models in a heterogeneous decentralized environment with offloading.<br>
    <br>
    <a href="https://pypi.org/project/bloombee/"><img src="https://img.shields.io/pypi/v/bloombee.svg?label=PyPI&color=green"></a>
    <a href="https://github.com/ai-decentralized/bloombee/actions"><img src="https://img.shields.io/github/actions/workflow/status/ai-decentralized/bloombee/pylint.yml?branch=main&label=Build"></a>
    <a href="https://discord.gg/Ypexx2rxt9"><img src="https://img.shields.io/discord/1267714065166241813?label=Discord&logo=discord&logoColor=white"></a>
    <a href="https://github.com/ai-decentralized/bloombee/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ai-decentralized/bloombee?color=blue"></a>
    <a href="https://pypi.org/project/bloombee/"><img src="https://img.shields.io/pypi/pyversions/bloombee"></a>
    <a href="https://pypi.org/project/bloombee/"><img src="https://img.shields.io/pypi/dm/bloombee?label=Downloads"></a>
    <a href="https://github.com/ai-decentralized/bloombee"><img src="https://img.shields.io/github/stars/ai-decentralized/bloombee?style=social"></a>
</p>

The rapid rise of generative AI has boosted demand for large language model (LLM) inference and fine-tuning services. While proprietary models are still favored, advancements in open-source LLMs have made them competitive. However, high costs and limited GPU resources hinder deployment. BloomBee is a decentralized offline serving system that leverages idle GPU resources to provide cost-effective access to LLMs.

Instead of requiring a single powerful machine, BloomBee splits a model's transformer blocks across multiple peers in a P2P network. If your GPU can only hold a small portion of a large model like LLaMA 3.1 (405B), you can join a network of servers each hosting different layers and collaboratively serve inference or fine-tuning requests.

<p align="center">
    🚀 &nbsp;<b><a href="https://colab.research.google.com/drive/1BZn0KrEGaNA2dlzmCTtTIjJKx3bNzOMs#scrollTo=1Qhi4I2PSGgg">Try now in Colab</a></b>
</p>

## Table of Contents

- [How It Works](#how-it-works)
- [Supported Models](#supported-models)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Start a Bootstrap Node](#1-start-a-bootstrap-node)
  - [2. Start Worker Servers](#2-start-worker-servers)
  - [3. Run Inference or Fine-tuning](#3-run-inference-or-fine-tuning)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Benchmarking](#benchmarking)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## How It Works

BloomBee distributes a model's transformer layers across a peer-to-peer network:

```
┌─────────────────────────────────────────────────────────┐
│  CLIENT (you)                                           │
│  • Runs word embeddings and the LM head locally         │
│  • Routes through remote layers via DHT                 │
└──────────────────────┬──────────────────────────────────┘
                       │ P2P (libp2p)
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Worker A │  │ Worker B │  │ Worker C │
   │ Layers   │  │ Layers   │  │ Layers   │
   │  0 – 15  │  │ 16 – 31  │  │ 32 – 47  │
   └──────────┘  └──────────┘  └──────────┘
         Peers registered in DHT
```

A **DHT (Distributed Hash Table)** keeps track of which server hosts which layers. The client automatically discovers and routes through available peers. Servers are decentralized — anyone with a compatible GPU can join and contribute capacity.

---

## Supported Models

| Model Family | Example HuggingFace IDs |
|---|---|
| **LLaMA / LLaMA 2 / LLaMA 3** | `meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B` |
| **BLOOM** | `bigscience/bloom-7b1`, `bigscience/bloom` |
| **Falcon** | `tiiuae/falcon-7b`, `tiiuae/falcon-40b` |
| **Mixtral** | `mistralai/Mixtral-8x7B-v0.1` |

Any HuggingFace model with a matching architecture can be served. Use `AutoDistributedModelForCausalLM` to load a model automatically.

---

## Prerequisites

- **Python** 3.8 or later
- **PyTorch** 1.12 or later (with CUDA for GPU support)
- A GPU with at least **~4 GB VRAM** to serve a portion of a model (workers)
- A machine with internet access and, optionally, a public IP or port forwarding for external peer discovery

> **Note:** The client machine does not need a GPU — only worker servers do.

---

## Installation

#### From PyPI
```bash
pip install bloombee
```

#### From Source
```bash
git clone https://github.com/ai-decentralized/BloomBee.git
cd BloomBee
pip install .
```

---

## Quick Start

([Try in Colab](https://colab.research.google.com/drive/1pENMOEoEV01DqBImZzuX_4jTV3fNwNga#scrollTo=oyCFDemCZsRs))

### 1. Start a Bootstrap Node

A bootstrap node is a lightweight DHT peer that helps other nodes discover each other. Start one first:

```bash
python -m bloombee.cli.run_dht \
  --host_maddrs /ip4/0.0.0.0/tcp/31340 \
  --identity_path bootstrap.id
```

You will see a line like:

```
Mon 00 01:23:45.678 [INFO] Running a DHT instance. To connect other peers to this one, use:
  --initial_peers /ip4/YOUR_IP/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ
```

Copy this address — you'll pass it as `--initial_peers` to all workers and clients.

> If you want your swarm accessible from outside your local network, make sure you have a **public IP address** or have **port forwarding** configured correctly.

### 2. Start Worker Servers

Export the bootstrap address for convenience:

```bash
export BBSERVER=/ip4/YOUR_IP/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ
```

Start workers, each hosting a slice of the model. For a 32-layer model, you might split it across two servers:

```bash
# Worker 1: hosts 16 transformer layers
python -m bloombee.cli.run_server meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --num_blocks 16 \
  --identity_path worker_1.id

# Worker 2: hosts the remaining 16 layers
python -m bloombee.cli.run_server meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --num_blocks 16 \
  --identity_path worker_2.id
```

Workers will automatically download their assigned model layers from HuggingFace on first run.

### 3. Run Inference or Fine-tuning

#### Inference

```bash
python benchmarks/benchmark_inference.py \
  --model meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --torch_dtype float32 \
  --seq_len 128
```

#### Fine-tuning

```bash
python benchmarks/benchmark_training.py \
  --model meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --torch_dtype float32 \
  --n_steps 20 \
  --batch_size 32 \
  --seq_len 128
```

---

## CLI Reference

### `bloombee.cli.run_dht` — Bootstrap Node

Starts a lightweight DHT peer for peer discovery. Does not load any model.

| Argument | Default | Description |
|---|---|---|
| `--host_maddrs` | — | Multiaddresses to listen on (e.g. `/ip4/0.0.0.0/tcp/31340`) |
| `--identity_path` | — | Path to store/load this node's persistent identity key |
| `--announce_maddrs` | — | Public multiaddresses to announce to other peers (useful behind NAT) |

### `bloombee.cli.run_server` — Worker Server

Loads and serves transformer blocks on a peer in the swarm.

| Argument | Default | Description |
|---|---|---|
| `model` | *(required)* | HuggingFace model name or local path |
| `--initial_peers` | — | Multiaddresses of bootstrap nodes to connect to |
| `--num_blocks` | auto | Number of transformer layers to serve |
| `--block_indices` | auto | Specific layer index range to serve (e.g. `0:16`) |
| `--identity_path` | — | Path to store/load this peer's persistent identity key |
| `--quant_type` | none | Quantization: `int8` (LLM.int8) or `nf4` (QLoRA 4-bit) |
| `--torch_dtype` | auto | Model precision: `float32`, `float16`, `bfloat16` |
| `--throughput` | `auto` | Reported throughput in tokens/sec; use `eval` to measure or `dry_run` to skip |
| `--cache_dir` | — | Directory to cache downloaded model weights |
| `--max_batch_size` | 2048 | Maximum number of tokens per forward batch |

---

## Python API

BloomBee integrates with HuggingFace Transformers. Use the `Auto` classes to load a distributed model:

```python
from transformers import AutoTokenizer
from bloombee import AutoDistributedModelForCausalLM

model = AutoDistributedModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    initial_peers=["/ip4/YOUR_IP/tcp/31340/p2p/Qm..."],
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

inputs = tokenizer("The quick brown fox", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

For efficient multi-turn generation, reuse an inference session to avoid reprocessing past tokens:

```python
with model.transformer.h.inference_session(max_length=512) as sess:
    for _ in range(20):
        outputs = model.generate(max_new_tokens=1, session=sess)
```

Available auto classes:

| Class | Use case |
|---|---|
| `AutoDistributedModelForCausalLM` | Text generation |
| `AutoDistributedModelForSequenceClassification` | Classification / fine-tuning |
| `AutoDistributedModel` | Raw transformer (no LM head) |

---

## Benchmarking

Three benchmark scripts are provided in `benchmarks/`:

```bash
# Measure autoregressive inference throughput (tokens/sec)
python benchmarks/benchmark_inference.py \
  --model meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --seq_len 256

# Measure forward pass throughput (tokens/sec)
python benchmarks/benchmark_forward.py \
  --model meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --batch_size 4 \
  --seq_len 128

# Measure training (forward + backward) throughput
python benchmarks/benchmark_training.py \
  --model meta-llama/Llama-2-7b-hf \
  --initial_peers $BBSERVER \
  --batch_size 4 \
  --seq_len 128 \
  --n_steps 20
```

---

## Examples

Jupyter notebook examples are in the `examples/` directory:

| Notebook | Description |
|---|---|
| [prompt-tuning-sst2.ipynb](examples/prompt-tuning-sst2.ipynb) | Prompt-tune LLaMA for sentiment classification (SST-2) |
| [prompt-tuning-personachat.ipynb](examples/prompt-tuning-personachat.ipynb) | Prompt-tune BLOOM for dialogue generation (PersonaChat) |

---

## Troubleshooting

**Workers cannot find each other**
- Ensure all workers use the same `--initial_peers` address.
- If running across machines, verify the bootstrap node has a public IP and the port is open.
- Use `--announce_maddrs` to explicitly advertise your public address if behind NAT.

**`ModuleNotFoundError: No module named 'bloombee'`**
- Run `pip install bloombee` or `pip install -e .` from the repository root.

**Out of GPU memory on a worker**
- Reduce `--num_blocks` to serve fewer layers.
- Enable quantization: `--quant_type int8` or `--quant_type nf4`.
- Use a smaller `--max_batch_size`.

**`transformers` version mismatch**
- BloomBee requires `transformers>=4.43.1,<4.44.0`. Install the pinned version:
  ```bash
  pip install "transformers>=4.43.1,<4.44.0"
  ```

**Slow inference / high latency**
- Latency increases with the number of network hops between layers. Place workers on the same local network when possible.
- Ensure workers report accurate throughput: use `--throughput eval` on the first run.

---

## Acknowledgements

BloomBee is built upon a few popular libraries:

  - [Hivemind](https://github.com/learning-at-home/hivemind) - A PyTorch library for decentralized deep learning across the Internet.
  - [FlexLLMGen](https://github.com/FMInference/FlexLLMGen) - An offloading-based system running on weak GPUs.
  - [Petals](https://github.com/bigscience-workshop/petals) - A library for decentralized LLMs fine-tuning and inference without offloading.