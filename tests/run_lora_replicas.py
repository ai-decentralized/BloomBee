"""Data parallelism over replicated spans on a real private swarm (CUDA).

Two (or more) workers each hold the whole tiny model. `train_step` spreads microbatches over the
replicas; with replica_sync="peer" the client uploads adapters to one replica, the others pull them
from it, and one replica sums the replicas' gradients. Gradients must match HuggingFace PEFT on the
whole batch, for both sync modes and across an optimizer update (a new adapter version).

Run from the checkout:  PYTHONPATH=src python tests/run_lora_replicas.py --output /tmp/lora_replicas
"""

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
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig as BBLoRAConfig
from bloombee.data_structures import ServerState
from bloombee.utils.dht import get_remote_module_infos
from bloombee.utils.hivemind_compat import DHT
from bloombee.utils.lora import PROJECTIONS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_lora_rpc import save_tiny_tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This integration test requires a CUDA GPU")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    base_dir = args.output / "base"
    torch.manual_seed(173)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
    )  # fmt: skip
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).half().save_pretrained(base_dir)
    save_tiny_tokenizer(base_dir)
    prefix = "lorarep_" + uuid.uuid4().hex[:12]
    dht = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False)
    peers = [str(a) for a in dht.get_visible_maddrs()]
    processes, logs, managers = [], [], []
    try:
        for index in range(args.replicas):
            log = (args.output / f"worker-{index}.log").open("w")
            logs.append(log)
            env = {
                **os.environ, "BLOOMBEE_ENABLE_LORA": "1", "BLOOMBEE_ENABLE_SPEC_PRUNER": "0",
                "BLOOMBEE_ENABLE_MICROBATCH_PIPELINE": "0", "BLOOMBEE_LOSSLESS_WRAPPER": "0",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
            }  # fmt: skip
            command = [
                sys.executable, "-m", "bloombee.cli.run_server", str(base_dir.resolve()),
                "--block_indices", "0:2", "--dht_prefix", prefix, "--num_handlers", "1", "--batch_size", "2",
                "--torch_dtype", "float16", "--throughput", "100", "--update_period", "1", "--expiration", "30",
                "--inference_max_length", "64", "--attn_cache_tokens", "128", "--max_batch_size", "512",
                "--balance_quality", "0", "--no_auto_relay", "--host_maddrs", "/ip4/127.0.0.1/tcp/0",
                "--skip_reachability_check", "--cache_gpu_percent", "100", "--w_gpu_percent", "100",
                "--initial_peers", *peers,
            ]  # fmt: skip
            processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        deadline, uids = time.monotonic() + 240, [f"{prefix}.{i}" for i in range(2)]
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Worker exited; inspect worker-*.log")
            infos = get_remote_module_infos(dht, uids, latest=True)
            online = [sum(s.state == ServerState.ONLINE for s in item.servers.values()) for item in infos]
            if all(o >= args.replicas for o in online):
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("Private swarm startup timed out")
        print("Workers ONLINE", flush=True)

        def load_client():
            model = AutoDistributedModelForCausalLM.from_pretrained(
                str(base_dir.resolve()), dht_prefix=prefix, initial_peers=peers, torch_dtype=torch.float16,
                connect_timeout=30, request_timeout=60, max_retries=1, update_period=1,
            ).cuda()  # fmt: skip
            seq = next(m for m in model.modules() if m.__class__.__name__ == "RemoteSequential")
            managers.append(seq.sequence_manager)
            return model

        ids = torch.tensor([[1, 5 + i, 9, 13, 17, 21 + i, 25, 29] for i in range(4)], device="cuda")
        report = {}
        for mode in ("peer", "client"):
            model = DistributedLoRA(
                load_client(), BBLoRAConfig(r=4, lora_alpha=8, target_modules=PROJECTIONS), replica_sync=mode
            ).cuda()
            adapter_dir = args.output / f"adapter-{mode}"
            model.save_pretrained(adapter_dir)
            reference = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float16, attn_implementation="eager").cuda(),
                adapter_dir, is_trainable=True,
            )  # fmt: skip
            for name, parameter in reference.named_parameters():
                if ".lora_" in name:
                    parameter.data = parameter.data.float()
            model.train()
            reference.train()
            params = [p for p in model.parameters() if p.requires_grad]
            ref_params = dict(reference.named_parameters())
            state = model._adapter_state()
            used = set()
            sequence = model.remote_layers
            original = sequence.acquire_route

            def recording(*a, **k):
                route = original(*a, **k)
                used.update(str(s.peer_id) for s in route)
                return route

            sequence.acquire_route = recording
            ref_opt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.3)
            opt = torch.optim.SGD(params, lr=0.3)
            errors = []
            for round_ in range(3):  # round 1 and 2 use updated adapters: a new version is pulled each time
                opt.zero_grad(set_to_none=True)
                ref_opt.zero_grad(set_to_none=True)
                loss = model.train_step(ids, labels=ids, micro_batches=4)
                expected = reference(ids, labels=ids, use_cache=False)
                expected.loss.backward()
                assert abs(loss - float(expected.loss)) < 2e-3, (mode, round_, loss, float(expected.loss))
                for key, parameter in state.items():
                    ref_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                        ".lora_B.weight", ".lora_B.default.weight"
                    )
                    target = ref_params[ref_key]
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    errors.append(float((parameter.grad - target.grad).abs().max()))
                    torch.testing.assert_close(parameter.grad, target.grad, rtol=0.08, atol=5e-5)
                opt.step()
                ref_opt.step()
            assert len(used) == args.replicas, f"expected microbatches on all replicas, used {used}"
            report[mode] = {"max_gradient_error": max(errors), "replicas_used": len(used)}
            sequence.acquire_route = original
            print(mode, report[mode], flush=True)
        for log in logs:
            log.flush()
        text = "".join((args.output / f"worker-{i}.log").read_text() for i in range(args.replicas))
        report["peer_pulls_logged"] = text.count("[LORA_PEER] pulled adapters")
        report["peer_reductions_logged"] = text.count("[LORA_PEER] reduced gradients")
        assert report["peer_pulls_logged"] >= 3, "peer mode never pulled adapters from the seed replica"
        assert report["peer_reductions_logged"] >= 3, "peer mode never reduced gradients on a leader replica"
        report["status"] = "passed"
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for manager in managers:
            try:
                manager.shutdown()
            except Exception:
                pass
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        for log in logs:
            log.close()
        dht.shutdown()
        os._exit(0 if (args.output / "report.json").exists() else 1)


if __name__ == "__main__":
    main()
