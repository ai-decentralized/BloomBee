"""Reproducible request workloads for benchmark_inference.py."""

import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.util import Finalize
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, __version__ as transformers_version

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP

_MODEL = None
_OPTIONS = None


def add_workload_arguments(parser):
    group = parser.add_argument_group("Independent requests (opt in; defaults preserve the legacy benchmark)")
    group.add_argument(
        "--dataset", type=Path, help="Local JSON/JSONL: prompt records, messages, or ShareGPT conversations"
    )
    group.add_argument("--prompt", help="Explicit prompt; use an empty string to start from BOS/EOS only")
    group.add_argument("--num_requests", type=int, help="Measured independent requests (default: 200 in request mode)")
    group.add_argument("--concurrency", type=int, default=1, help="Independent client processes in request mode")
    group.add_argument(
        "--request_rate", type=float, help="Offered requests/s with seeded Poisson arrivals; default: closed loop"
    )
    group.add_argument(
        "--warmup_requests", type=int, default=10, help="Complete warmup requests per client, excluded from metrics"
    )
    group.add_argument("--seed", type=int, default=42, help="Dataset sampling and arrival schedule seed")
    group.add_argument("--revision", help="Model/tokenizer revision; HF models are pinned to their resolved commit")
    group.add_argument(
        "--max_input_tokens", type=int, help="Filter longer dataset inputs; never pad or truncate real prompts"
    )
    group.add_argument(
        "--ignore_eos", action="store_true", help="Generate the full output budget rather than stopping at EOS"
    )
    group.add_argument(
        "--rpc_timeout", type=float, default=120, help="RPC timeout; request mode disables automatic retries"
    )
    group.add_argument("--startup_timeout", type=float, default=600, help="Client loading and warmup barrier timeout")
    group.add_argument(
        "--result_path", type=Path, default=Path("inference-results.json"), help="Summary and per-request results"
    )


def read_dataset(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()] if path.suffix == ".jsonl" else json.load(f)
    if not isinstance(rows, list):
        raise ValueError("Dataset must be a JSON list or JSONL records")
    return rows


def encode_record(row, tokenizer, output_cap):
    reference = None
    if "conversations" in row:
        roles = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
        messages = [{"role": roles[turn["from"]], "content": turn["value"]} for turn in row["conversations"]]
    elif "messages" in row:
        messages = list(row["messages"])
    else:
        prompt = row["prompt"]
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"] if prompt else []
        messages = None
    if messages is not None:
        if messages and messages[-1]["role"] == "assistant":
            reference = messages.pop()["content"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Messages must end with a user turn, optionally followed by a reference assistant answer")
        if tokenizer.chat_template:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        elif len(messages) == 1 and messages[0]["role"] == "user":
            ids = tokenizer(messages[0]["content"], add_special_tokens=True)["input_ids"]
        else:
            raise ValueError(
                "Multi-turn messages require a tokenizer chat template; use prompt records for base models"
            )
    if not ids:
        start = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if start is None:
            raise ValueError("Empty prompts require a tokenizer BOS or EOS token")
        ids = [start]
    output_len = row.get("output_len")
    if output_len is None:
        output_len = len(tokenizer(reference, add_special_tokens=False)["input_ids"]) if reference else output_cap
    if not isinstance(output_len, int) or isinstance(output_len, bool) or output_len < 1:
        raise ValueError("output_len must be a positive integer")
    return {"input_ids": ids, "output_len": min(output_len, output_cap)}


def prepare_workload(args, tokenizer):
    count = args.num_requests if args.num_requests is not None else 200
    wanted = count + args.warmup_requests
    if args.dataset is not None:
        rows = list(enumerate(read_dataset(args.dataset)))
        random.Random(args.seed).shuffle(rows)
    else:
        rows = [
            (
                i,
                {
                    "prompt": args.prompt
                    if args.prompt is not None
                    else args.prompt_template.format(i=args.prompt_start_index + i)
                },
            )
            for i in range(wanted)
        ]
    selected = []
    filtered = {"too_long": 0, "invalid": 0, "invalid_examples": []}
    for index, row in rows:
        try:
            item = encode_record(row, tokenizer, args.seq_len)
        except (KeyError, TypeError, ValueError) as error:
            if args.dataset is None:
                raise
            filtered["invalid"] += 1
            if len(filtered["invalid_examples"]) < 5:
                filtered["invalid_examples"].append({"source_index": index, "error": str(error)})
            continue
        if args.max_input_tokens is not None and len(item["input_ids"]) > args.max_input_tokens:
            filtered["too_long"] += 1
            continue
        selected.append({"source_index": index, **item})
        if len(selected) == wanted:
            break
    if len(selected) < wanted:
        raise ValueError(
            f"Need {wanted} eligible records including warmup; found {len(selected)}. "
            f"Rejected: {filtered}. No oversampling is performed."
        )
    rng = random.Random(args.seed + 1)
    arrival = 0.0
    measured = []
    for i, item in enumerate(selected[args.warmup_requests :]):
        if i and args.request_rate is not None:
            arrival += rng.expovariate(args.request_rate)
        measured.append({"request_id": i, "arrival_offset_s": arrival, **item})
    return selected[: args.warmup_requests], measured, filtered


def run_request(item, arrival):
    started = perf_counter()
    result = {
        "request_id": item.get("request_id"),
        "source_index": item["source_index"],
        "input_tokens": len(item["input_ids"]),
        "output_budget": item["output_len"],
        "arrival_s": arrival,
        "started_s": started,
        "client_pid": os.getpid(),
    }
    times, tokens = [], []
    try:
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=_OPTIONS.client_device)
        eos = _MODEL.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        with torch.inference_mode(), _MODEL.inference_session(max_length=ids.shape[1] + item["output_len"]) as session:
            for step in range(item["output_len"]):
                output = _MODEL.generate(ids if step == 0 else None, max_new_tokens=1, session=session, do_sample=False)
                token = int(output[0, -1].item())
                times.append(perf_counter())
                tokens.append(token)
                if not _OPTIONS.ignore_eos and token in eos:
                    break
        result.update(success=True, stop_reason="eos" if tokens[-1] in eos and not _OPTIONS.ignore_eos else "length")
    except Exception as error:
        result.update(success=False, error=f"{type(error).__name__}: {error}")
    result.update(output_tokens=len(tokens), output_ids=tokens, token_times_s=times, finished_s=perf_counter())
    return result


def init_client(args, warmup, barrier):
    global _MODEL, _OPTIONS
    _OPTIONS = args
    torch.manual_seed(args.seed)
    kwargs = dict(
        initial_peers=args.initial_peers,
        torch_dtype=DTYPE_MAP[args.torch_dtype],
        use_server_to_server=args.use_server_to_server,
        revision=args.revision,
        max_retries=1,
        request_timeout=args.rpc_timeout,
        connect_timeout=args.rpc_timeout,
    )
    if args.dht_prefix is not None:
        kwargs["dht_prefix"] = args.dht_prefix
    _MODEL = AutoDistributedModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.client_device).eval()
    Finalize(None, _MODEL.transformer.h.sequence_manager.shutdown, exitpriority=10)
    for item in warmup:
        result = run_request(item, perf_counter())
        if not result["success"]:
            raise RuntimeError("Warmup failed: " + result["error"])
    barrier.wait(timeout=args.startup_timeout)


def client_ready():
    return os.getpid()


def summarize(results, duration, offered):
    successful = [row for row in results if row["success"]]
    values = {key: [] for key in ("ttft_ms", "service_ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "queue_ms")}
    for row in successful:
        times = row["token_times_s"]
        row.update(
            ttft_ms=(times[0] - row["arrival_s"]) * 1000,
            service_ttft_ms=(times[0] - row["started_s"]) * 1000,
            e2e_ms=(times[-1] - row["arrival_s"]) * 1000,
            queue_ms=(row["started_s"] - row["arrival_s"]) * 1000,
        )
        for key in ("ttft_ms", "service_ttft_ms", "e2e_ms", "queue_ms"):
            values[key].append(row[key])
        values["itl_ms"].extend(np.diff(times) * 1000)
        row["tpot_ms"] = None
        if len(times) > 1:
            row["tpot_ms"] = (times[-1] - times[0]) * 1000 / (len(times) - 1)
            values["tpot_ms"].append(row["tpot_ms"])
    metrics = {}
    for key, samples in values.items():
        metrics[key] = {
            "count": len(samples),
            **(
                {"mean": float(np.mean(samples)), **{f"p{p}": float(np.percentile(samples, p)) for p in (50, 95, 99)}}
                if len(samples)
                else {}
            ),
        }
    return {
        "offered_requests": offered,
        "completed_requests": len(successful),
        "failed_requests": sum(not row["success"] for row in results),
        "uncompleted_requests": offered - len(results),
        "duration_s": duration,
        "output_tokens": sum(row["output_tokens"] for row in successful),
        "requests_per_s": len(successful) / duration if duration else 0,
        "output_tokens_per_s": sum(row["output_tokens"] for row in successful) / duration if duration else 0,
        "failure_rate": (offered - len(successful)) / offered,
        "metrics": metrics,
    }


def benchmark_requests(args):
    if args.dataset is not None and args.prompt is not None:
        raise ValueError("Choose --dataset or --prompt")
    if args.batch_size != 1 or args.n_processes != 1 or args.prompt_len is not None:
        raise ValueError(
            "Request mode uses --concurrency with batch_size=1; --prompt_len and --n_processes are legacy options"
        )
    if (
        args.concurrency < 1
        or args.warmup_requests < 0
        or args.seq_len < 1
        or (args.num_requests is not None and args.num_requests < 1)
    ):
        raise ValueError("concurrency, num_requests and seq_len must be positive; warmup_requests must be nonnegative")
    if args.request_rate is not None and (not np.isfinite(args.request_rate) or args.request_rate <= 0):
        raise ValueError("request_rate must be finite and positive")
    if (
        args.rpc_timeout <= 0
        or args.startup_timeout <= 0
        or (args.max_input_tokens is not None and args.max_input_tokens < 1)
    ):
        raise ValueError("Timeouts and max_input_tokens must be positive")
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    args.revision = getattr(config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=False)
    warmup, requests, filtered = prepare_workload(args, tokenizer)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_hash = None
    if args.dataset is not None:
        digest = hashlib.sha256()
        with args.dataset.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        dataset_hash = digest.hexdigest()
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "dataset": str(args.dataset) if args.dataset else None,
        "dataset_sha256": dataset_hash,
        "tokenizer": type(tokenizer).__name__,
        "chat_template": tokenizer.chat_template,
        "filtered_inputs": filtered,
        "warmup_per_client": warmup,
        "requests": requests,
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "timing": "Pretokenized inputs; TTFT/E2E include offered-load queueing; token timestamps include generation/sampling, not text decoding. Each request has a fresh KV session.",
    }
    manifest_path = args.result_path.with_suffix(".workload.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.concurrency)
    results, active = [], {}
    began = ended = None
    failure = None
    try:
        with ProcessPoolExecutor(
            max_workers=args.concurrency, mp_context=ctx, initializer=init_client, initargs=(args, warmup, barrier)
        ) as pool:
            ready = [pool.submit(client_ready) for _ in range(args.concurrency)]
            for future in ready:
                future.result(timeout=args.startup_timeout)
            began = perf_counter()
            index = 0
            last_progress = 0
            while index < len(requests) or active:
                now = perf_counter()
                while index < len(requests) and len(active) < args.concurrency:
                    item = requests[index]
                    arrival = began + item["arrival_offset_s"] if args.request_rate is not None else now
                    if arrival > now:
                        break
                    active[pool.submit(run_request, item, arrival)] = item
                    index += 1
                delay = 0.05
                if index < len(requests) and args.request_rate is not None and len(active) < args.concurrency:
                    delay = min(delay, max(0.0, began + requests[index]["arrival_offset_s"] - perf_counter()))
                if not active:
                    sleep(delay)
                    continue
                done, _ = wait(active, timeout=delay, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    del active[future]
                if len(results) - last_progress >= 25 or len(results) == len(requests):
                    print(
                        f"[REQUEST_PROGRESS] finished={len(results)}/{len(requests)} "
                        f"failed={sum(not row['success'] for row in results)}",
                        flush=True,
                    )
                    last_progress = len(results)
            ended = perf_counter()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    duration = (ended or perf_counter()) - began if began is not None else 0.0
    results.sort(key=lambda row: row["request_id"])
    summary = summarize(results, duration, len(requests))
    report = {
        "workload": str(manifest_path),
        "summary": summary,
        "requests": results,
        "error": failure,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": transformers_version,
        },
    }
    args.result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {args.result_path}", flush=True)
    if failure or summary["failure_rate"]:
        raise RuntimeError(failure or "Some benchmark requests failed; see " + str(args.result_path))
