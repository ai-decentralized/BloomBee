#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import load_dataset
from hivemind.utils.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local source tree is used instead of an older installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SRC_STR = str(_SRC_ROOT)
if _SRC_STR in sys.path:
    sys.path.remove(_SRC_STR)
sys.path.insert(0, _SRC_STR)

from bloombee import AutoDistributedSpeculativeModel
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
from bloombee.models.llama.spec_decoding_drafter import MultiSSMDrafter

logger = get_logger()

# ── 固定超参 ──────────────────────────────────────────────
BATCH_SIZE     = 1    # 固定 batch size
NUM_PROMPTS    = 256  # 从数据集中采样的 prompt 总数
MAX_NEW_TOKENS = 128  # 每个 prompt 推理的 token 数
# ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of benchmark iterations")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    pipe_recv, pipe_send = mp.Pipe(duplex=False)
    processes = [mp.Process(target=benchmark_inference, args=(i, args, pipe_send)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    speed = np.mean([pipe_recv.recv() for _ in range(args.n_processes)])
    logger.info(f"Final result: {speed=:.2f} tokens/s")


@torch.inference_mode()
def benchmark_inference(process_idx, args, result_pipe):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drafter = MultiSSMDrafter(
        ssm_model_name="JackFram/llama-68m",
        num_workers=1,
        device="cuda"
    )
    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 数据集：随机采样 NUM_PROMPTS 条 ──────────────────
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    indices = random.sample(range(len(dataset)), NUM_PROMPTS)
    test_prompts = [dataset[i]["instruction"] for i in indices]
    logger.info(f"Sampled {NUM_PROMPTS} prompts from dataset")

    # ── 逐条推理（batch_size=1）───────────────────────────
    total_generated = 0
    total_time      = 0.0
    speeds          = []

    for sample_idx, prompt in enumerate(test_prompts):
        # tokenize 单条 prompt
        input_ids = tokenizer(
            [prompt], return_tensors="pt", padding=True
        ).to(device)["input_ids"]                          # shape: (1, seq_len)

        start_time = perf_counter()
        output = model.generate(
            input_ids=input_ids,
            drafter=drafter,
            max_new_tokens=MAX_NEW_TOKENS,
        )                                                  # shape: (1, seq_len + new_tokens)
        elapsed = perf_counter() - start_time

        # 计算本条实际生成的 token 数
        prompt_len    = input_ids.shape[1]
        output_len    = output.shape[1]
        generated_num = output_len - prompt_len

        sample_speed = generated_num / elapsed if elapsed > 0 else 0.0
        speeds.append(sample_speed)
        total_generated += generated_num
        total_time      += elapsed

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(
            f"[{sample_idx + 1:>3}/{NUM_PROMPTS}] "
            f"generated={generated_num} tokens, "
            f"time={elapsed:.3f}s, "
            f"speed={sample_speed:.2f} tok/s"
        )
        logger.info(f"  Prompt : {prompt}")
        logger.info(f"  Output : {decoded}")

    # ── 汇总统计 ─────────────────────────────────────────
    avg_speed = total_generated / total_time if total_time > 0 else 0.0
    logger.info("=" * 60)
    logger.info(f"Process {process_idx} summary")
    logger.info(f"  Prompts processed : {NUM_PROMPTS}")
    logger.info(f"  Batch size        : {BATCH_SIZE}")
    logger.info(f"  Max new tokens    : {MAX_NEW_TOKENS}")
    logger.info(f"  Total generated   : {total_generated} tokens")
    logger.info(f"  Total time        : {total_time:.2f}s")
    logger.info(f"  Avg speed         : {avg_speed:.2f} tokens/s")
    logger.info(f"  Per-sample speeds : min={min(speeds):.2f}, "
                f"max={max(speeds):.2f}, "
                f"mean={np.mean(speeds):.2f} tok/s")
    logger.info("=" * 60)

    result_pipe.send(avg_speed)


if __name__ == "__main__":
    main()