#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import random
import sys
import json
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
    parser.add_argument(
        "--prompt_start_index",
        type=int,
        default=1,
        help="Starting index for prompt generation; default=1 avoids the degenerate 'Number 0' case",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Number {i}: ",
        help="Prompt template. Must contain '{i}', e.g. 'Number {i}: ' or 'Topic {i}: '",
    )
    parser.add_argument("--group_idx", type=int, default=0, 
                    help="Which group to run (0-9)")
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
        ssm_model_name="JackFram/llama-160m",
        num_workers=1,
        device="cuda"
    )
    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    batch_size = getattr(args, 'batch_size', 32)
    group_idx = getattr(args, 'group_idx', 0)
    
    # 加载固定的prompt组
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    with open("eval_indices.json", "r") as f:
        groups = json.load(f)
    
    indices = groups[group_idx]
    sampled = dataset.select(indices)
    test_prompts = [item["instruction"] for item in sampled]
    
    logger.info(f"Running group {group_idx}/{len(groups)-1}")
    logger.info(f"Prompts: {test_prompts}")
    
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(
        test_prompts, 
        return_tensors="pt", 
        padding=True
    ).to(device)["input_ids"]
    
    max_new_tokens = getattr(args, 'seq_len', 128)
    
    # warmup
    logger.info("Warming up...")
    _ = model.generate(
        input_ids=input_ids, 
        drafter=drafter, 
        max_new_tokens=10
    )
    
    # 正式计时
    logger.info("Starting benchmark...")
    start_time = perf_counter()
    result = model.generate(
        input_ids=input_ids, 
        drafter=drafter, 
        max_new_tokens=max_new_tokens
    )
    elapsed_time = perf_counter() - start_time
    
    original_output_ids = result
    
    total_generated = 128 * batch_size
    throughput = total_generated / elapsed_time
    
    logger.info(f"Group {group_idx} | "
                f"Total time: {elapsed_time:.4f}s | "
                f"Throughput: {throughput:.2f} tokens/s | "
                f"Generated tokens per sample: {total_generated}")
    
    # 保存结果
    result_label = "pruned" if getattr(args, 'pruning', False) else "unpruned"
    output_file = f"results_{result_label}_group_{group_idx}.json"
    result_data = {
        "group_idx": group_idx,
        "pruning": getattr(args, 'pruning', False),
        "throughput": throughput,
        "elapsed_time": elapsed_time,
        "total_generated": total_generated,
        "generated_tokens_nums": total_generated,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    result_pipe.send(throughput)


if __name__ == "__main__":
    main()
