#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of benchmark iterations")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (reserved)")
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
    logger.info(f"Final result: {speed=:.2f}")


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
    
    batch_size = getattr(args, 'batch_size', 8)
    base_prompt = (
        "Quantum mechanics explains the behavior of particles at very small scales. "
        "Neural networks learn patterns by adjusting weights through backpropagation. "
        "Distributed systems require robust consensus mechanisms to maintain state. "
        "Optimization algorithms like gradient descent are fundamental to machine learning. "
        "Transformer architectures rely on attention mechanisms to capture dependencies. "
        "Reinforcement learning optimizes actions by maximizing cumulative rewards. "
        "Bayesian inference updates beliefs based on observed evidence and prior knowledge. "
        "Convex optimization problems guarantee global minima under certain conditions. "
        "Signal processing extracts meaningful information from noisy measurements. "
    )
    prompts = [
        f"{base_prompt} Example {i + 1} discusses large-scale AI systems and scientific discovery."
        for i in range(batch_size)
    ]
    test_prompts = prompts

    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(test_prompts, return_tensors="pt", padding=True).to(device)["input_ids"]

    result = ""
    start_time = perf_counter()
    max_new_tokens = getattr(args, 'seq_len', 128)
    result = model.generate(input_ids=input_ids, drafter=drafter, max_new_tokens=max_new_tokens)
    time = perf_counter() - start_time
    generated_tokens_nums = []
    for i in range(batch_size):
        prompt_mask = input_ids[i].ne(tokenizer.pad_token_id)
        prompt_length = prompt_mask.sum().item()
        result_mask = result[i].ne(tokenizer.pad_token_id) & result[i].ne(0)
        result_length = result_mask.sum().item()
        generated_tokens_num = result_length - prompt_length
        generated_tokens_nums.append(generated_tokens_num)
        
        logger.info(f"result: {result[i]}")
    
    avg_generated_tokens = sum(generated_tokens_nums) / batch_size
    speed = avg_generated_tokens / time

    decoded_results = tokenizer.batch_decode(result, skip_special_tokens=True)

    logger.info(f"benchmark_inference batch size: {batch_size}")
    logger.info(f"Total time: {time:.4f}s, Average speed: {speed:.2f} tokens/s")
    logger.info(f"Generated tokens per sample: {generated_tokens_nums}")

    for i, (prompt, decoded_result) in enumerate(zip(test_prompts, decoded_results)):
        logger.info(f"Sample {i}:")
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Result: {decoded_result}")
        logger.info(f"  Generated tokens: {generated_tokens_nums[i]}")
    
    
    result_pipe.send(speed)


if __name__ == "__main__":
    main()
