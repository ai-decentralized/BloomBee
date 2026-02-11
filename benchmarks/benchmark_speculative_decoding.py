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

    ssm = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m")
    ssm = ssm.to(device).eval()
    # warm up ssm to reduce inference later
    with torch.no_grad():
        dummy_input = torch.ones(1, 8, dtype=torch.long, device=device)
        ssm(dummy_input, attention_mask=torch.ones_like(dummy_input))

    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 配置参数
    batch_size = getattr(args, 'batch_size', 10)
    num_iterations = getattr(args, 'num_iterations', 1)

    # 加载数据集
    dataset = load_dataset("tatsu-lab/alpaca")["train"]

    # 统计信息
    all_speeds = []
    all_times = []
    all_generated_tokens = []

    for iteration in range(num_iterations):
        logger.info(f"=" * 50)
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info(f"=" * 50)

        # 随机采样 batch_size 条数据
        indices = random.sample(range(len(dataset)), batch_size)
        sampled = dataset.select(indices)
        test_prompts = [item["instruction"] for item in sampled]

        # Tokenize
        input_ids = tokenizer(test_prompts, return_tensors="pt", padding=True).to(device)["input_ids"]

        # 推理
        start_time = perf_counter()
        result = model.generate(input_ids=input_ids, ssm=ssm)
        time_elapsed = perf_counter() - start_time

        # 计算生成的 token 数
        generated_tokens_nums = []
        for i in range(batch_size):
            prompt_length = input_ids[i].ne(tokenizer.pad_token_id).sum().item()
            result_length = result[i].ne(tokenizer.pad_token_id).sum().item()
            generated_tokens_num = result_length - prompt_length
            generated_tokens_nums.append(generated_tokens_num)

        avg_generated_tokens = sum(generated_tokens_nums) / batch_size
        speed = avg_generated_tokens / time_elapsed

        all_speeds.append(speed)
        all_times.append(time_elapsed)
        all_generated_tokens.extend(generated_tokens_nums)

        # 解码结果
        decoded_results = tokenizer.batch_decode(result, skip_special_tokens=True)

        # 打印本次迭代结果
        logger.info(f"Iteration {iteration + 1} - Time: {time_elapsed:.4f}s, Speed: {speed:.2f} tokens/s")
        logger.info(f"Generated tokens per sample: {generated_tokens_nums}")

        for i, (prompt, decoded_result) in enumerate(zip(test_prompts, decoded_results)):
            logger.info(f"  Sample {i}:")
            logger.info(f"    Prompt: {prompt[:100]}...")  # 截断显示
            logger.info(f"    Result: {decoded_result[:200]}...")  # 截断显示
            logger.info(f"    Generated tokens: {generated_tokens_nums[i]}")

    # 汇总统计
    logger.info(f"=" * 50)
    logger.info(f"Summary ({num_iterations} iterations, batch_size={batch_size})")
    logger.info(f"=" * 50)
    logger.info(f"Average speed: {sum(all_speeds) / len(all_speeds):.2f} tokens/s")
    logger.info(f"Min speed: {min(all_speeds):.2f} tokens/s")
    logger.info(f"Max speed: {max(all_speeds):.2f} tokens/s")
    logger.info(f"Average time per batch: {sum(all_times) / len(all_times):.4f}s")
    logger.info(f"Total samples processed: {num_iterations * batch_size}")
    logger.info(f"Average tokens generated per sample: {sum(all_generated_tokens) / len(all_generated_tokens):.2f}")

    # 返回平均速度
    avg_speed = sum(all_speeds) / len(all_speeds)
    result_pipe.send(avg_speed)


if __name__ == "__main__":
    main()
