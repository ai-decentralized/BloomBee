"""Short real-checkpoint LoRA smoke test; no downloads or task-quality claims."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig
from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--weight-gpu-percent", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    ids = tokenizer("The capital of France is Paris. The capital of Italy is Rome.", return_tensors="pt")["input_ids"][
        :, :24
    ]
    prefix = "lora_smoke_" + uuid.uuid4().hex[:10]
    dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
    peers = [str(addr) for addr in dht.get_visible_maddrs()]
    processes, logs, managers = [], [], []
    report = dict(
        model=str(args.model),
        layers=config.num_hidden_layers,
        hidden=config.hidden_size,
        stages=2,
        steps=args.steps,
        weight_gpu_percent=args.weight_gpu_percent,
        status="running",
    )
    try:
        split = config.num_hidden_layers // 2
        for i, (start, end) in enumerate(((0, split), (split, config.num_hidden_layers))):
            log = (args.output / f"worker-{i}.log").open("w")
            logs.append(log)
            command = [
                sys.executable,
                "-m",
                "bloombee.cli.run_server",
                str(args.model.resolve()),
                "--block_indices",
                f"{start}:{end}",
                "--dht_prefix",
                prefix,
                "--num_handlers",
                "1",
                "--batch_size",
                "1",
                "--torch_dtype",
                "float16",
                "--throughput",
                "100",
                "--update_period",
                "5",
                "--expiration",
                "60",
                "--inference_max_length",
                "64",
                "--attn_cache_tokens",
                "128",
                "--max_batch_size",
                "128",
                "--balance_quality",
                "0",
                "--no_auto_relay",
                "--host_maddrs",
                "/ip4/127.0.0.1/tcp/0",
                "--skip_reachability_check",
                "--w_gpu_percent",
                str(args.weight_gpu_percent),
                "--initial_peers",
                *peers,
            ]
            env = {
                **os.environ,
                "BLOOMBEE_ENABLE_LORA": "1",
                "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
            }
            processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Worker exited; inspect worker logs")
            infos = get_remote_module_infos(
                dht, [f"{prefix}.{i}" for i in range(config.num_hidden_layers)], latest=True
            )
            if all(any(info.state == ServerState.ONLINE for info in item.servers.values()) for item in infos):
                break
            time.sleep(1)
        else:
            raise TimeoutError("Pretrained workers did not become online")
        print("Pretrained workers ONLINE", flush=True)

        def load():
            base = AutoDistributedModelForCausalLM.from_pretrained(
                str(args.model.resolve()),
                initial_peers=peers,
                dht_prefix=prefix,
                torch_dtype=torch.float16,
                connect_timeout=30,
                request_timeout=300,
                max_retries=1,
            ).cuda()
            managers.append(base.model.layers.sequence_manager)
            return base

        ids = ids.cuda()
        model = DistributedLoRA(load(), LoRAConfig(r=8, lora_alpha=16)).cuda().train()
        params = [p for p in model.parameters() if p.requires_grad]
        report["trainable_parameters"] = sum(p.numel() for p in params)
        report["input_ids"] = ids.tolist()
        optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0)
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            output = model(ids, labels=ids)
            output.loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            if step == 0:
                assert all(p.grad.count_nonzero() == 0 for p in params[::2])
                report["first_B_gradient_norm"] = sum(float(p.grad.norm()) for p in params[1::2])
                assert report["first_B_gradient_norm"] > 0
            if step > 0:
                assert any(p.grad.norm() > 0 for p in params[::2])
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(output.loss))
            print(f"pretrained step={step} loss={losses[-1]:.6f}", flush=True)
        model.eval()
        with torch.no_grad():
            final = model(ids, labels=ids)
        report.update(losses=losses, initial_loss=losses[0], final_loss=float(final.loss))
        assert report["final_loss"] < losses[0]
        model.save_pretrained(args.output / "adapter")
        torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
        reloaded = DistributedLoRA.from_pretrained(load(), args.output / "adapter").cuda().eval()
        with torch.no_grad():
            reload_logits = reloaded(ids).logits
        torch.testing.assert_close(final.logits, reload_logits, rtol=0, atol=0)
        report["adapter_roundtrip_exact"] = True
        report["route"] = [
            {"peer": str(s.peer_id), "start": s.start, "end": s.end}
            for s in model.remote_layers.sequence_manager.make_sequence(mode="max_throughput")
        ]
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        for manager in managers:
            manager.shutdown()
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
        dht.shutdown()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
