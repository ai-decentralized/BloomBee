#!/usr/bin/env python3
import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from bloombee import AutoDistributedSpeculativeModel
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
from bloombee.models.llama.spec_decoding_drafter import MultiSSMDrafter

from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
import random

logger = get_logger()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
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
        num_workers=2,
        device="cuda"
    )
    # warm up ssm to reduce inference later
    # with torch.no_grad():
    #     dummy_input = torch.ones(1, 8, dtype=torch.long, device=device)
    #     ssm(dummy_input, attention_mask=torch.ones_like(dummy_input))

    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    batch_size = 8
    dataset = load_dataset("tatsu-lab/alpaca")["train"]
    indices = random.sample(range(len(dataset)), batch_size)
    sampled = dataset.select(indices)
    test_prompts = []
    # for item in sampled:
        # test_prompts.append(item["instruction"])
        
    test_prompts.append("Hi,")
    test_prompts.append("")
    test_prompts.append("")
    test_prompts.append("")
    test_prompts.append("")
    test_prompts.append("")
    test_prompts.append("")
    test_prompts.append("")

    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    input_ids = tokenizer(test_prompts, return_tensors="pt", padding=True).to(device)["input_ids"]
    
    # test_prompt = ""
    # bos_token_id = tokenizer.bos_token_id
    # if bos_token_id is not None:
    #     input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    # else:
    #     # 如果tokenizer没有bos_token_id，可能需要手动获取或处理
    #     logger.warning("Tokenizer does not have a bos_token_id. Using an empty tensor.")
    #     input_ids = torch.tensor([[]], dtype=torch.long, device=device)
    

    result = ""
    start_time = perf_counter()
    result = model.generate(input_ids=input_ids, drafter=drafter)
    time = perf_counter() - start_time
    generated_tokens_nums = []
    for i in range(batch_size):
        prompt_length = input_ids[i].ne(tokenizer.pad_token_id).sum().item()
        result_length = result[i].ne(tokenizer.pad_token_id).sum().item()
        generated_tokens_num = result_length - prompt_length
        generated_tokens_nums.append(generated_tokens_num)
        logger.info(f"result: {result[i]}")
    
    avg_generated_tokens = sum(generated_tokens_nums) / batch_size
    speed = avg_generated_tokens / time

    # 解码所有结果
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