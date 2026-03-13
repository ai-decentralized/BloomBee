#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter
import logging

import numpy as np
import torch
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()
_TIMING_KEYS = (
    "t_gpu2cpu_ms",
    "t_cpu2nic_ms",
    "t_nic2nic_ms",
    "t_nic2cpu_ms",
    "t_cpu2gpu_ms",
    "inference_latency_ms",
    "throughput_tok_s",
    "comm_volume_bytes",
    "t_gpu_compute_ms",
    "net_latency_ms",
    "net_bandwidth_gbps",
)

# Set logging level to INFO to see all debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _build_timing_table(session_summary, *, inference_latency_ms: float, throughput_tok_s: float):
    summary = {key: 0.0 for key in _TIMING_KEYS}
    if isinstance(session_summary, dict):
        for key in summary:
            summary[key] = float(session_summary.get(key, 0.0))
    summary["inference_latency_ms"] = float(inference_latency_ms)
    summary["throughput_tok_s"] = float(throughput_tok_s)
    return summary


def _format_timing_table_line(summary, *, label: str) -> str:
    comm_volume_mb = float(summary.get("comm_volume_bytes", 0.0)) / (1024.0 * 1024.0)
    return (
        f"[TIMING_TABLE] {label} "
        f"T_GPU->CPU={summary['t_gpu2cpu_ms']:.2f}ms "
        f"T_CPU->NIC={summary['t_cpu2nic_ms']:.2f}ms "
        f"T_NIC->NIC={summary['t_nic2nic_ms']:.2f}ms "
        f"T_NIC->CPU={summary['t_nic2cpu_ms']:.2f}ms "
        f"T_CPU->GPU={summary['t_cpu2gpu_ms']:.2f}ms "
        f"InferenceLatency={summary['inference_latency_ms']:.2f}ms "
        f"Throughput={summary['throughput_tok_s']:.2f}tok/s "
        f"CommunicateVolume={comm_volume_mb:.4f}MB "
        f"T_GPU_Compute={summary['t_gpu_compute_ms']:.2f}ms "
        f"NetLatency={summary['net_latency_ms']:.2f}ms "
        f"NetBandwidth={summary['net_bandwidth_gbps']:.4f}Gbps"
    )


def _average_timing_tables(tables):
    valid_tables = [table for table in tables if isinstance(table, dict)]
    if not valid_tables:
        return None
    return {
        key: float(np.mean([table.get(key, 0.0) for table in valid_tables]))
        for key in _TIMING_KEYS
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Number of tokens to generate (generation length)")
    parser.add_argument("--prompt_len", type=int, default=None, help="Desired prompt/prefill length in tokens (optional)")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Client batch size (number of sequences to generate in parallel)")
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
    parser.add_argument(
        "--token_log_every",
        type=int,
        default=8,
        help="Log generated tokens every N decode steps (<=0 logs only first/last step)",
    )
    parser.add_argument(
        "--log_all_tokens",
        action="store_true",
        help="Log generated tokens for every decode step (high-volume logs)",
    )
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

    results = [pipe_recv.recv() for _ in range(args.n_processes)]
    speeds = [r[0] for r in results]
    effective_speeds = [r[1] for r in results]
    timing_tables = [r[2] for r in results]
    
    avg_speed = np.mean(speeds)
    avg_effective_throughput = np.mean(effective_speeds)
    logger.info(f"Final result: throughput={avg_speed:.2f} tokens/sec/sequence, effective_throughput={avg_effective_throughput:.2f} tokens/sec")
    avg_timing_table = _average_timing_tables(timing_tables)
    if avg_timing_table is not None:
        timing_line = _format_timing_table_line(avg_timing_table, label=f"Process=avg n={len(timing_tables)}")
        logger.info(timing_line)
        print(timing_line, flush=True)


@torch.inference_mode()
def benchmark_inference(process_idx, args, result_pipe):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # Using use_fast=False since LlamaTokenizerFast takes a long time to start, and we decode 1 token at a time anyway
    
    # Set pad_token for LLaMA tokenizer (required for batch padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype],
        use_server_to_server=True  # Explicitly enable server-to-server communication
    ) 
    logger.info(f"Created model: {process_idx=} {model.device=}")

    # Prepare batch of prompts for benchmarking
    batch_size = getattr(args, 'batch_size', 1)
    
    # Use different prompts for each batch item to verify micro-batch correctness.
    # NOTE: Starting from 0 may trigger degenerate greedy output ("0000...") for some models.
    prompt_indices = [args.prompt_start_index + i for i in range(batch_size)]
    if "{i}" not in args.prompt_template:
        raise ValueError("--prompt_template must include '{i}' placeholder")
    prompts = [args.prompt_template.format(i=i) for i in prompt_indices]
    
    if args.prompt_len is None:
        encodings = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = encodings["input_ids"]
    else:
        target_prompt_length = args.prompt_len
        bos_token_id = tokenizer.bos_token_id
        filler_sentence = (
            " Advanced research explores interdisciplinary insights, collaborative innovation, "
            "scientific computation, trustworthy deployment, and sustainable engineering practices."
        )
        filler_tokens = tokenizer(filler_sentence, add_special_tokens=False)["input_ids"]
        if not filler_tokens:
            filler_tokens = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
        processed = []
        for prompt in prompts:
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if bos_token_id is not None:
                full_tokens = [bos_token_id] + prompt_tokens
            else:
                full_tokens = prompt_tokens[:]
            if len(full_tokens) >= target_prompt_length:
                full_tokens = full_tokens[:target_prompt_length]
            else:
                while len(full_tokens) < target_prompt_length:
                    need = target_prompt_length - len(full_tokens)
                    full_tokens.extend(filler_tokens[:need])
            processed.append(full_tokens)
        input_ids = torch.tensor(processed, dtype=torch.long)
    
    logger.info(f"{process_idx=} Client batch_size={batch_size}, input_ids.shape={input_ids.shape}")
    for i, prompt in enumerate(prompts):
        logger.info(
            f"{process_idx=} batch[{i}] prompt: '{prompt}' "
            f"(prompt_index={prompt_indices[i]}, token_ids={input_ids[i].tolist()})"
        )
    temp_result_tokens = input_ids
    
    # Calculate max_length: prompt_length + number of tokens to generate
    prompt_length = input_ids.shape[1]
    if args.prompt_len is not None:
        target_prompt_length = args.prompt_len
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if target_prompt_length < prompt_length:
            input_ids = input_ids[:, :target_prompt_length]
        elif target_prompt_length > prompt_length:
            extra = target_prompt_length - prompt_length
            pad_block = torch.full((batch_size, extra), pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_block], dim=1)
        prompt_length = target_prompt_length
        temp_result_tokens = input_ids
        logger.info(f"{process_idx=} adjusted prompt_length to {prompt_length} tokens")

    total_max_length = prompt_length + args.seq_len
    logger.info(f"{process_idx=} prompt_length={prompt_length}, generating {args.seq_len} tokens, total_max_length={total_max_length}")
    
    step_times = []
    step_latencies = []  # Track individual step latencies for cross-GPU analysis
    cross_gpu_latencies = []  # Track cross-GPU transfer latencies
    server_processing_latencies = []  # Track server processing latencies
    
    with model.transformer.h.inference_session(max_length=total_max_length) as sess:
        logger.info(f"{process_idx=} Created inference session with max_length={total_max_length}")
        logger.info(f"[BENCHMARK_START] Process={process_idx} | BatchSize={batch_size} | SeqLen={args.seq_len}")
        
        for step in range(args.seq_len):
            step_start_time = perf_counter()
            
            # For the first step, pass input_ids; for subsequent steps, generate() will use session state
            if step == 0:
                logger.info(f"{process_idx=} {step=} First step, passing input_ids.shape={input_ids.shape}")
                outputs = model.generate(input_ids, max_new_tokens=1, session=sess)
            else:
                logger.debug(f"{process_idx=} {step=} Subsequent step, using session state")
                outputs = model.generate(max_new_tokens=1, session=sess)
            
            step_end_time = perf_counter()
            step_latency_ms = (step_end_time - step_start_time) * 1000
            step_latencies.append(step_latency_ms)
            
            logger.debug(f"{process_idx=} {step=} After generate, outputs.shape={outputs.shape}")
            
            log_step_tokens = (
                args.log_all_tokens
                or step < 2
                or step == args.seq_len - 1
                or (args.token_log_every > 0 and step % args.token_log_every == 0)
            )
            token_line = ""
            if log_step_tokens:
                token_ids = [outputs[b][-1].item() for b in range(outputs.shape[0])]
                token_texts = [tokenizer.decode([tid]) for tid in token_ids]
                parts = []
                i = 0
                while i < len(token_texts):
                    txt = token_texts[i]
                    count = 1
                    while i + count < len(token_texts) and token_texts[i + count] == txt:
                        count += 1
                    display = repr(txt)
                    parts.append(f"{display}x{count}" if count > 1 else display)
                    i += count
                token_line = " | tokens=[" + " ".join(parts) + "]"
            
            temp_result_tokens = torch.cat([temp_result_tokens, outputs[:, -1:]], dim=1)

            if step >= args.warmup_steps:
                step_times.append(perf_counter() - step_start_time)
                speed = 1 / np.mean(step_times)
                effective_speed = speed * batch_size
                logger.info(
                    f"[STEP] P{process_idx} step={step} | {step_latency_ms:.1f}ms | "
                    f"{speed:.2f} tok/s/seq ({effective_speed:.1f} tok/s){token_line}"
                )
                cross_gpu_latencies.append(step_latency_ms)
                server_processing_latencies.append(step_latency_ms)
            else:
                logger.info(
                    f"[STEP] P{process_idx} step={step} | {step_latency_ms:.1f}ms | "
                    f"(warmup){token_line}"
                )
        
        # Calculate and log statistics
        warmup_latencies = step_latencies[args.warmup_steps:]
        warmup_cross_gpu_latencies = cross_gpu_latencies[args.warmup_steps:]
        warmup_server_processing_latencies = server_processing_latencies[args.warmup_steps:]
        
        if warmup_latencies:
            mean_latency = np.mean(warmup_latencies)
            median_latency = np.median(warmup_latencies)
            p95_latency = np.percentile(warmup_latencies, 95)
            p99_latency = np.percentile(warmup_latencies, 99)
            min_latency = np.min(warmup_latencies)
            max_latency = np.max(warmup_latencies)
            
            # Cross-GPU Transfer Latency statistics
            if warmup_cross_gpu_latencies:
                cross_gpu_mean = np.mean(warmup_cross_gpu_latencies)
                cross_gpu_median = np.median(warmup_cross_gpu_latencies)
                cross_gpu_p95 = np.percentile(warmup_cross_gpu_latencies, 95)
                cross_gpu_p99 = np.percentile(warmup_cross_gpu_latencies, 99)
            
            # Server Processing Latency statistics
            if warmup_server_processing_latencies:
                server_mean = np.mean(warmup_server_processing_latencies)
                server_median = np.median(warmup_server_processing_latencies)
                server_p95 = np.percentile(warmup_server_processing_latencies, 95)
                server_p99 = np.percentile(warmup_server_processing_latencies, 99)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"[PERFORMANCE_SUMMARY] Process={process_idx}")
            logger.info(f"{'='*80}")
            
            # Overall Latency Summary
            logger.info(f"[OVERALL_LATENCY]")
            logger.info(f"  Mean:   {mean_latency:.2f}ms")
            logger.info(f"  Median: {median_latency:.2f}ms")
            logger.info(f"  P95:    {p95_latency:.2f}ms")
            logger.info(f"  P99:    {p99_latency:.2f}ms")
            logger.info(f"  Min:    {min_latency:.2f}ms")
            logger.info(f"  Max:    {max_latency:.2f}ms")
            
            # Cross-GPU Transfer Latency Summary
            if warmup_cross_gpu_latencies:
                logger.info(f"\n[CROSS_GPU_TRANSFER_LATENCY]")
                logger.info(f"  Mean:   {cross_gpu_mean:.2f}ms")
                logger.info(f"  Median: {cross_gpu_median:.2f}ms")
                logger.info(f"  P95:    {cross_gpu_p95:.2f}ms")
                logger.info(f"  P99:    {cross_gpu_p99:.2f}ms")
            
            # Server Processing Latency Summary
            if warmup_server_processing_latencies:
                logger.info(f"\n[SERVER_PROCESSING_LATENCY]")
                logger.info(f"  Mean:   {server_mean:.2f}ms")
                logger.info(f"  Median: {server_median:.2f}ms")
                logger.info(f"  P95:    {server_p95:.2f}ms")
                logger.info(f"  P99:    {server_p99:.2f}ms")
            
            logger.info(f"{'='*80}\n")
    
    # Calculate final throughput and effective throughput
    if step_times:
        speed = 1 / np.mean(step_times)
    else:
        # Fallback if no steps were measured (shouldn't happen in normal operation)
        speed = 0.0
    effective_speed = speed * batch_size
    mean_latency_ms = float(np.mean(warmup_latencies)) if warmup_latencies else 0.0
    session_timing_summary = getattr(sess, "timing_summary", None)
    timing_table = _build_timing_table(
        session_timing_summary,
        inference_latency_ms=mean_latency_ms,
        throughput_tok_s=effective_speed,
    )
    timing_line = _format_timing_table_line(timing_table, label=f"Process={process_idx}")
    logger.info(timing_line)
    print(timing_line, flush=True)
    
    # Show final generated text for each batch
    logger.info(f"\n{'='*80}")
    logger.info(f"[FINAL RESULTS] {process_idx=}")
    logger.info(f"{'='*80}")
    for batch_idx in range(temp_result_tokens.shape[0]):
        full_text = tokenizer.decode(temp_result_tokens[batch_idx], skip_special_tokens=True)
        logger.info(f"\nbatch[{batch_idx}] Full generated text:\n{full_text}\n")
    logger.info(f"{'='*80}\n")
    
    result_pipe.send((speed, effective_speed, timing_table))


if __name__ == "__main__":
    main()
