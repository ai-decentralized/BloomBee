"""LoRA fine-tuning benchmark with an accuracy gate.

Starts a private DHT and local workers on the GPUs you name, trains LoRA through BloomBee
and reports throughput, a time and bytes breakdown, GPU memory and the loss curve. With
--reference it then trains the same adapter on the same batches with HuggingFace + PEFT on
one GPU and compares the loss curves and the held-out loss (the accuracy gate).

  # real model, 2 pipeline stages, PEFT parity
  python benchmarks/benchmark_lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --gpus 0,1 \
      --batch 4 --seq 512 --steps 30 --reference --output out/tinyllama

  # random checkpoint (nothing downloaded), simulated 50 ms / 200 Mbit/s links
  python benchmarks/benchmark_lora.py --random 8,2048,32000 --gpus 0,1 --sim-link 50,200 --output out/wan

Training data: --data JSONL with {"text": ...}; otherwise this repository's own docs and
source code (deterministic, no download). A random checkpoint trains on random tokens.

Workers are only launched on GPUs that look idle; the check repeats before each launch.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def nvidia_smi(query, section="gpu"):
    out = subprocess.run(
        ["nvidia-smi", f"--query-{section}={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[cell.strip() for cell in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def assert_gpus_idle(gpus, max_mem_mib=1000, max_util=10, attempts=4):
    """Refuse to use a GPU that anyone else is using (retries: utilization lags a finished job)."""
    for attempt in range(attempts):
        try:
            return _check_idle(gpus, max_mem_mib, max_util)
        except RuntimeError:
            if attempt == attempts - 1:
                raise
            time.sleep(8)


def _check_idle(gpus, max_mem_mib, max_util):
    rows = {int(r[0]): r for r in nvidia_smi("index,uuid,memory.used,utilization.gpu")}
    busy = {r[0] for r in nvidia_smi("gpu_uuid,pid", section="compute-apps")}
    for gpu in gpus:
        if gpu not in rows:
            raise RuntimeError(f"GPU {gpu} does not exist on this machine")
        _, uuid_, mem, util = rows[gpu]
        if int(mem) > max_mem_mib or int(util) > max_util or uuid_ in busy:
            raise RuntimeError(f"GPU {gpu} is not idle (mem={mem} MiB, util={util}%, busy={uuid_ in busy})")


# Fix the client's visible GPU and the simulated link before torch / bloombee are imported.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpus", default="0")
_pre.add_argument("--client-gpu", type=int, default=None)
_pre.add_argument("--sim-link", default="")
_pre.add_argument("--skip-idle-check", action="store_true")
_known, _ = _pre.parse_known_args()
_GPUS = [int(g) for g in _known.gpus.split(",")]
_CLIENT_GPU = _known.client_gpu if _known.client_gpu is not None else _GPUS[0]
if "--help" not in sys.argv and "-h" not in sys.argv:
    if not _known.skip_idle_check:
        assert_gpus_idle(sorted(set(_GPUS + [_CLIENT_GPU])))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_CLIENT_GPU)
    if _known.sim_link:
        os.environ["BLOOMBEE_SIM_LINK"] = _known.sim_link

import torch  # noqa: E402

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig  # noqa: E402
from bloombee.data_structures import ServerState  # noqa: E402
from bloombee.utils import sim_link  # noqa: E402
from bloombee.utils.dht import get_remote_module_infos  # noqa: E402
from bloombee.utils.hivemind_compat import DHT  # noqa: E402


def make_random_checkpoint(path, layers, hidden, vocab, seq):
    from transformers import LlamaConfig, LlamaForCausalLM

    marker = path / "bench_config.json"
    spec = dict(layers=layers, hidden=hidden, vocab=vocab)
    if marker.exists() and json.loads(marker.read_text()) == spec:
        return
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(173)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 11 // 4 // 64 * 64,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 128,
        num_key_value_heads=max(1, hidden // 512),
        max_position_embeddings=max(4096, seq * 2),
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).half().save_pretrained(path)
    marker.write_text(json.dumps(spec))


def build_dataset(args, vocab):
    """Return (train, eval) LongTensors of shape [n, seq]."""
    generator = torch.Generator().manual_seed(args.seed)
    needed = args.batch * (args.warmup + args.steps) + args.batch * args.eval_batches
    if args.random:
        data = torch.randint(0, vocab, (needed, args.seq), generator=generator)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.data:
            texts = [json.loads(line)["text"] for line in args.data.read_text().splitlines() if line.strip()]
        else:
            files = sorted(REPO.glob("*.md")) + sorted((REPO / "src").rglob("*.py"))
            texts = [f.read_text(errors="ignore") for f in files]
        tokens = []
        for text in texts:
            tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
        count = len(tokens) // args.seq
        if count < needed:
            raise ValueError(f"Corpus has {count} sequences of {args.seq}; need {needed}")
        chunks = torch.tensor(tokens[: count * args.seq]).view(count, args.seq)
        data = chunks[torch.randperm(count, generator=generator)[:needed]]
    split = args.batch * (args.warmup + args.steps)
    return data[:split], data[split:]


class Swarm:
    def __init__(self, args, model_path, out, layers):
        self.args, self.model_path, self.out, self.layers = args, model_path, out, layers
        self.prefix = "lorabench_" + uuid.uuid4().hex[:10]
        self.dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
        self.peers = [str(a) for a in self.dht.get_visible_maddrs()]
        self.processes, self.logs = [], []

    def plan(self):
        if self.args.topology == "full":
            return [(gpu, 0, self.layers) for gpu in _GPUS[: self.args.replicas]]
        cuts = [round(i * self.layers / len(_GPUS)) for i in range(len(_GPUS) + 1)]
        return [(gpu, cuts[i], cuts[i + 1]) for i, gpu in enumerate(_GPUS)]

    def start(self):
        plan = self.plan()
        checked = set()
        for index, (gpu, start, end) in enumerate(plan):
            if gpu not in checked and not self.args.skip_idle_check and gpu != _CLIENT_GPU:
                assert_gpus_idle([gpu])
            checked.add(gpu)
            log = (self.out / f"worker-{index}.log").open("w")
            self.logs.append(log)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "BLOOMBEE_ENABLE_LORA": "1",
                "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
                "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE": "0",
                "BLOOMBEE_LOSSLESS_WRAPPER": "0",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                **dict(kv.split("=", 1) for kv in self.args.worker_env),
            }
            command = [
                sys.executable, "-m", "bloombee.cli.run_server", str(self.model_path),
                "--block_indices", f"{start}:{end}",
                "--dht_prefix", self.prefix,
                "--num_handlers", "1",
                "--batch_size", "2",
                "--torch_dtype", "float16",
                "--throughput", "100",
                "--update_period", "1",
                "--expiration", "30",
                "--inference_max_length", "64",
                "--attn_cache_tokens", "128",
                "--max_batch_size", "1048576",
                "--balance_quality", "0",
                "--no_auto_relay",
                "--host_maddrs", "/ip4/127.0.0.1/tcp/0",
                "--skip_reachability_check",
                "--cache_gpu_percent", "100",
                "--w_gpu_percent", "100",
                "--initial_peers", *self.peers,
                *self.args.worker_args,
            ]  # fmt: skip
            self.processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        uids = [f"{self.prefix}.{i}" for i in range(self.layers)]
        expected = [sum(start <= i < end for _, start, end in plan) for i in range(self.layers)]
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in self.processes):
                raise RuntimeError(f"A worker exited early; see {self.out}/worker-*.log")
            infos = get_remote_module_infos(self.dht, uids, latest=True)
            online = [sum(s.state == ServerState.ONLINE for s in item.servers.values()) for item in infos]
            if all(o >= e for o, e in zip(online, expected)):
                return plan
            time.sleep(1)
        raise TimeoutError("Workers did not come online in 20 minutes")

    def stop(self):
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in self.processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in self.logs:
            log.close()
        self.dht.shutdown()


class GpuSampler(threading.Thread):
    """Mean utilisation and peak memory of the benchmark's own GPUs."""

    def __init__(self, gpus):
        super().__init__(daemon=True)
        self.util = {g: [] for g in gpus}
        self.mem = {g: 0 for g in gpus}
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                for index, util, mem in nvidia_smi("index,utilization.gpu,memory.used"):
                    index = int(index)
                    if index in self.util:
                        self.util[index].append(int(util))
                        self.mem[index] = max(self.mem[index], int(mem))
            except Exception:
                pass
            time.sleep(0.25)

    def finish(self):
        self.stop_flag.set()
        self.join()
        return (
            {g: round(statistics.mean(v), 1) if v else None for g, v in self.util.items()},
            dict(self.mem),
        )


def run_training(model, params, train, args, *, timed):
    """Identical loop for BloomBee and the PEFT reference. Returns per-step losses and times."""
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    losses, times, transfer = [], [], []
    micro = max(1, args.micro_batches)
    train_step = getattr(model, "train_step", None) if args.use_train_step else None
    for step, ids in enumerate(train.split(args.batch)):
        ids = ids.cuda()
        torch.cuda.synchronize()
        before = sim_link.snapshot()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if train_step is not None:
            loss = float(train_step(ids, labels=ids, micro_batches=micro))
        else:
            loss = 0.0
            for chunk in ids.split(max(1, len(ids) // micro)):
                out = model(input_ids=chunk, labels=chunk, use_cache=False)
                scale = len(chunk) / len(ids)
                (out.loss * scale).backward()
                loss += float(out.loss.detach()) * scale
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        after = sim_link.snapshot()
        losses.append(loss)
        if timed and step >= args.warmup:
            times.append(elapsed)
            transfer.append({k: after[k] - before[k] for k in after})
        if step % 5 == 0 or step == len(train) // args.batch - 1:
            print(f"step={step} loss={loss:.5f} time={elapsed:.3f}s", flush=True)
    return losses, times, transfer


@torch.no_grad()
def evaluate(model, data, batch):
    total, count = 0.0, 0
    for ids in data.split(batch):
        ids = ids.cuda()
        total += float(model(input_ids=ids, labels=ids, use_cache=False).loss) * len(ids)
        count += len(ids)
    return total / max(count, 1)


def run_reference(model_path, adapter_dir, train, held_out, args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    for name, parameter in model.named_parameters():
        if ".lora_" in name:
            parameter.data = parameter.data.float()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    losses, _, _ = run_training(model, params, train, args, timed=False)
    model.eval()
    return losses, evaluate(model, held_out, args.batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="comma-separated idle GPUs hosting workers")
    parser.add_argument("--client-gpu", type=int, default=None, help="client GPU (default: first of --gpus)")
    parser.add_argument("--skip-idle-check", action="store_true")
    parser.add_argument("--topology", choices=("split", "full"), default="split")
    parser.add_argument("--replicas", type=int, default=1, help="full topology: worker replicas")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="HF id or local path of a LLaMA/Qwen3 checkpoint")
    source.add_argument("--random", help="random LLaMA checkpoint: layers,hidden,vocab")
    parser.add_argument("--data", type=Path, help='JSONL {"text": ...}; default: this repo\'s text')
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--micro-batches", dest="micro_batches", type=int, default=1)
    parser.add_argument("--use-train-step", action="store_true", help="use DistributedLoRA.train_step if present")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--eval-batches", dest="eval_batches", type=int, default=4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boundary-compression", dest="boundary_compression", default="none",
                        help="opt-in lossy activation codec: none, int8, hadamard-int8")
    parser.add_argument("--adapter-wire", dest="adapter_wire", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--sim-link", default="", help="'<latency_ms>,<mbit>' simulated client link")
    parser.add_argument("--reference", action="store_true", help="also train HF+PEFT and compare (accuracy gate)")
    parser.add_argument("--worker-env", nargs="*", default=[], help="extra KEY=VALUE env for workers")
    parser.add_argument("--worker-args", nargs=argparse.REMAINDER, default=[], help="extra run_server args (last)")
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.random:
        layers, hidden, vocab = (int(x) for x in args.random.split(","))
        model_path = (args.output.parent / f"random-{layers}-{hidden}-{vocab}").resolve()
        make_random_checkpoint(model_path, layers, hidden, vocab, args.seq)
        model_path = str(model_path)
    else:
        from transformers import AutoConfig

        model_path = args.model
        config = AutoConfig.from_pretrained(model_path)
        layers, vocab = config.num_hidden_layers, config.vocab_size
    train, held_out = build_dataset(args, vocab)

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    report = dict(
        label=args.label,
        commit=git.stdout.strip() or ((REPO / ".commit").read_text().strip() if (REPO / ".commit").exists() else ""),
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        devices=[r[1] for r in nvidia_smi("index,name")],
        tokens_per_step=args.batch * args.seq,
        status="running",
    )
    swarm = Swarm(args, model_path, args.output, layers)
    try:
        plan = swarm.start()
        report["plan"] = [dict(gpu=g, blocks=f"{s}:{e}") for g, s, e in plan]
        print("workers online:", report["plan"], flush=True)
        client = AutoDistributedModelForCausalLM.from_pretrained(
            model_path,
            dht_prefix=swarm.prefix,
            initial_peers=swarm.peers,
            torch_dtype=torch.float16,
            connect_timeout=60,
            request_timeout=1200,
            max_retries=2,
            update_period=1,
        ).cuda()
        torch.manual_seed(args.seed)
        config = LoRAConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                            target_modules=tuple(args.targets.split(",")))  # fmt: skip
        extra = {}
        if args.boundary_compression != "none":
            extra["boundary_compression"] = args.boundary_compression
        if args.adapter_wire != "float32":
            extra["adapter_wire"] = args.adapter_wire
        lora = DistributedLoRA(client, config, **extra).cuda().train()
        adapter_dir = args.output / "initial-adapter"
        lora.save_pretrained(adapter_dir)
        params = [p for p in lora.parameters() if p.requires_grad]
        sampler = GpuSampler(sorted(set(_GPUS + [_CLIENT_GPU])))
        sampler.start()
        losses, times, transfer = run_training(lora, params, train, args, timed=True)
        util, peak_mem = sampler.finish()
        lora.eval()
        eval_loss = evaluate(lora, held_out, args.batch)
        median = statistics.median(times)
        report.update(
            step_time_s_median=round(median, 4),
            step_time_s_all=[round(t, 4) for t in times],
            tokens_per_s=round(args.batch * args.seq / median, 1),
            bytes_per_step=dict(
                sent=int(statistics.median(t["bytes_sent"] for t in transfer)),
                received=int(statistics.median(t["bytes_received"] for t in transfer)),
                calls=int(statistics.median(t["calls"] for t in transfer)),
            ),
            gpu_util_mean_pct=util,
            gpu_peak_mem_mib=peak_mem,
            losses=[round(x, 5) for x in losses],
            eval_loss=round(eval_loss, 5),
            trainable_parameters=sum(p.numel() for p in params),
        )
        lora.remote_layers.sequence_manager.shutdown()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        swarm.stop()
    if args.reference:
        del client, lora
        torch.cuda.empty_cache()
        ref_losses, ref_eval = run_reference(model_path, adapter_dir, train, held_out, args)
        diffs = [abs(a - b) for a, b in zip(losses, ref_losses)]
        report["reference"] = dict(
            losses=[round(x, 5) for x in ref_losses],
            eval_loss=round(ref_eval, 5),
            max_abs_loss_diff=round(max(diffs), 5),
            mean_abs_loss_diff=round(statistics.mean(diffs), 5),
            final_train_loss_rel_diff=round(abs(losses[-1] - ref_losses[-1]) / abs(ref_losses[-1]), 5),
            eval_loss_rel_diff=round(abs(eval_loss - ref_eval) / abs(ref_eval), 5),
        )
        report["accuracy_gate_passed"] = report["reference"]["eval_loss_rel_diff"] <= 0.005
    report["status"] = "done"
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ("label", "tokens_per_s", "step_time_s_median", "bytes_per_step", "eval_loss")}
    if "reference" in report:
        summary["reference"] = {k: v for k, v in report["reference"].items() if k != "losses"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
