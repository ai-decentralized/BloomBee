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
    
    batch_size = getattr(args, 'batch_size', 32)
    group_idx = getattr(args, 'group_idx', 0)
    num_train_iters = getattr(args, 'num_train_iters', 256)
    save_interval = getattr(args, 'save_interval', 32)
    
    # 加载数据集（只加载一次）
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    all_prompts = [item["instruction"] for item in dataset if item["instruction"].strip()]
    logger.info(f"Loaded {len(all_prompts)} prompts from dataset")
    
    tokenizer.pad_token = tokenizer.eos_token
    max_new_tokens = getattr(args, 'seq_len', 128)
    
    def sample_batch():
        """每次随机采样 batch_size 条 prompt 并 tokenize"""
        sampled_prompts = random.sample(all_prompts, batch_size)
        ids = tokenizer(
            sampled_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)["input_ids"]
        return sampled_prompts, ids
    
    # warmup
    logger.info("Warming up...")
    _, warmup_ids = sample_batch()
    with torch.inference_mode():
        _ = model.generate(
            input_ids=warmup_ids,
            drafter=drafter,
            max_new_tokens=10
        )
    
    # 训练循环
    logger.info(f"Starting training loop for {num_train_iters} iterations...")
    
    total_loss = 0.0
    total_throughput = 0.0
    train_stats = []
    
    for iteration in range(num_train_iters):
        # 每次随机采样新的一批 prompts
        current_prompts, input_ids = sample_batch()
        
        start_time = perf_counter()
        
        result = model.generate(
            input_ids=input_ids,
            drafter=drafter,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
        )
        
        elapsed_time = perf_counter() - start_time
        
        # train_step
        if hasattr(model, 'pruner') and model.pruner is not None:
            train_info = model.pruner.train_step(
                middle_hidden_states=result.middle_hidden_states,
                final_logits=result.final_logits,
                attention_mask=result.attention_mask,
                draft_tokens=result.draft_tokens,
            )
            loss = train_info['total_loss']
            total_loss += loss
            
            train_stats.append({
                "iteration": iteration,
                "loss": loss,
                "pos_count": train_info['pos_count'],
                "neg_count": train_info['neg_count'],
            })
            
            logger.info(
                f"[{iteration+1}/{num_train_iters}] "
                f"loss={loss:.4f} | "
                f"pos={train_info['pos_count']} neg={train_info['neg_count']} | "
                f"time={elapsed_time:.2f}s"
            )
        else:
            logger.warning(f"[{iteration+1}/{num_train_iters}] model.pruner not found, skipping train_step")
        
        # 定期保存
        if (iteration + 1) % save_interval == 0:
            ckpt_path = f"pruner_ckpt_iter{iteration+1}.pt"
            if hasattr(model, 'pruner') and model.pruner is not None:
                model.pruner.save_model(ckpt_path)
                logger.info(f"Checkpoint saved to {ckpt_path}")
    
    # 训练结束统计
    avg_loss = total_loss / num_train_iters
    logger.info("=" * 60)
    logger.info(f"Training complete | avg_loss={avg_loss:.4f}")
    logger.info("=" * 60)
    
    # 保存最终模型
    final_ckpt_path = f"pruner_final_group{group_idx}.pt"
    if hasattr(model, 'pruner') and model.pruner is not None:
        model.pruner.save_model(final_ckpt_path)
        logger.info(f"Final model saved to {final_ckpt_path}")
    
    # 保存训练统计
    result_label = "pruned" if getattr(args, 'pruning', False) else "unpruned"
    output_file = f"results_{result_label}_group_{group_idx}.json"
    result_data = {
        "group_idx": group_idx,
        "pruning": getattr(args, 'pruning', False),
        "avg_loss": avg_loss,
        "num_train_iters": num_train_iters,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "train_stats": train_stats,
    }
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    result_pipe.send(0)


if __name__ == "__main__":
    main()
