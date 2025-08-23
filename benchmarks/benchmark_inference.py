#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Using use_fast=False since LlamaTokenizerFast takes a long time to start, and we decode 1 token at a time anyway

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    step_times = []
    
    logger.info(f"🔍 [Process {process_idx}] BOS token id: {tokenizer.bos_token_id}")
    logger.info(f"🔍 [Process {process_idx}] Starting inference session...")
    # test_prompt = "Simply put, the theory of relativity states that"
    test_prompt =""
    result = ""
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=True)
    temp_result_tokens = input_ids
    
    with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
        

        for step in range(args.seq_len):
            start_time = perf_counter()

            logger.info(f"🔍 [Process {process_idx}] Step {step} - Before generation:")
            logger.info(f"🔍 [Process {process_idx}] Current result length: {len(result)}")
            logger.info(f"🔍 [Process {process_idx}] Current result text: {repr(result)}")
            # if (step == 0):
            #     outputs = model.generate(input_ids, max_new_tokens=1, session=sess)
            # else:
            #     outputs = model.generate(max_new_tokens=1, session=sess)
                
            outputs = model.generate(max_new_tokens=1, session=sess)    
                
            logger.info(f"🔍 [Process {process_idx}] Step {step} - After generation:")
            logger.info(f"🔍 [Process {process_idx}] Generated outputs shape: {outputs.shape}")
            logger.info(f"🔍 [Process {process_idx}] Generated outputs: {outputs}")
            logger.info(f"🔍 [Process {process_idx}] Full sequence: {outputs[0]}")
            
   
            new_token_id = outputs[0][-1].item()  
            logger.info(f"🔍 [Process {process_idx}] New token id: {new_token_id}")
            

            new_token_text = tokenizer.decode([new_token_id])
            logger.info(f"🔍 [Process {process_idx}] New token text: {repr(new_token_text)}")
            

            temp_result_tokens = torch.cat([temp_result_tokens, outputs[:, -1:]], dim=1)
            full_decoded = tokenizer.decode(temp_result_tokens[0])
            logger.info(f"🔍 [Process {process_idx}] temp_result: {repr(full_decoded)}")

            if step >= args.warmup_steps:
                step_times.append(perf_counter() - start_time)
                speed = 1 / np.mean(step_times)
                logger.info(f"{process_idx=} {step=} {speed=:.2f}")
                
            logger.info(f"🔍 [Process {process_idx}] Step {step} completed\n" + "="*50)
            
    logger.info(f"Generated text (process {process_idx}): {repr(result)}")
    logger.info(f"Generated text length: {len(result)} characters")
    result_pipe.send(speed)


if __name__ == "__main__":
    main()