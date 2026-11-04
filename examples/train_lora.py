"""Standard LoRA on BloomBee workers; client AdamW and explicit accumulation.

Input: local JSONL records {"text": "..."}. Concatenate with EOS, chunk into
unpadded fixed-length causal sequences. All workers need BLOOMBEE_ENABLE_LORA=1.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM, DistributedLoRA, LoRAConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial_peers", nargs="+", required=True)
    parser.add_argument("--dht_prefix")
    parser.add_argument("--data", type=Path, required=True, help='Local JSONL: {"text": "..."}')
    parser.add_argument("--output", type=Path, required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume_adapter", type=Path, help="Weights only; starts a fresh optimizer")
    resume.add_argument("--resume_checkpoint", type=Path, help="Restore adapter, optimizer, step and RNG")
    parser.add_argument("--checkpoint_every", type=int, default=0, help="Optimizer steps; 0 saves only at the end")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.gradient_accumulation_steps) < 1 or args.seq_len < 2:
        raise ValueError("steps, batch and accumulation must be positive; seq_len must be >=2")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = []
    for line in args.data.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tokens.extend(tokenizer(record["text"], add_special_tokens=True)["input_ids"])
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
    count = len(tokens) // args.seq_len
    if count < 1:
        raise ValueError("Dataset is shorter than one unpadded seq_len chunk")
    chunks = torch.tensor(tokens[: count * args.seq_len]).view(count, args.seq_len)
    training_config = dict(
        model=args.model,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        accumulation=args.gradient_accumulation_steps,
        lr=args.lr,
        tokens_sha256=hashlib.sha256(chunks.numpy().tobytes()).hexdigest(),
    )
    losses, start_step = [], 0
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if saved["metadata"]["training_config"] != training_config:
            raise ValueError("Resume requires the same model, tokenized data, device and training settings")
        start_step, losses = saved["step"], saved["metadata"]["losses"]
        if args.steps < start_step:
            raise ValueError("steps is the total target step count and cannot precede the checkpoint")
        del saved
    if (args.output / f"checkpoint-{args.steps}").exists():
        raise FileExistsError("Use a fresh output directory; final checkpoint already exists")
    kwargs = dict(
        initial_peers=args.initial_peers,
        torch_dtype=torch.float16,
        request_timeout=300,
        connect_timeout=30,
        max_retries=1,
    )
    if args.dht_prefix:
        kwargs["dht_prefix"] = args.dht_prefix
    base = AutoDistributedModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    if args.resume_adapter or args.resume_checkpoint:
        model = DistributedLoRA.from_pretrained(base, args.resume_adapter or args.resume_checkpoint)
    else:
        model = DistributedLoRA(
            base, LoRAConfig(args.r, args.lora_alpha, args.lora_dropout, tuple(args.target_modules))
        )
    model.to(args.device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    start = time.perf_counter()
    try:
        if args.resume_checkpoint:
            model.load_training_state(args.resume_checkpoint, optimizer)
        for step in range(start_step, args.steps):
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                ids = chunks[torch.randint(count, (args.batch_size,), device="cpu")].to(args.device)
                output = model(input_ids=ids, labels=ids, use_cache=False)
                (output.loss / args.gradient_accumulation_steps).backward()
                mean_loss += float(output.loss.detach()) / args.gradient_accumulation_steps
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(mean_loss)
            completed = step + 1
            print(f"step={completed} loss={mean_loss:.6f} grad_norm={float(norm):.6f}", flush=True)
            if completed == args.steps or (args.checkpoint_every and completed % args.checkpoint_every == 0):
                model.save_training_checkpoint(
                    args.output / f"checkpoint-{completed}",
                    optimizer,
                    step=completed,
                    metadata=dict(training_config=training_config, losses=losses),
                )
        if start_step == args.steps:
            model.save_training_checkpoint(
                args.output / f"checkpoint-{start_step}",
                optimizer,
                step=start_step,
                metadata=dict(training_config=training_config, losses=losses),
            )
        model.save_pretrained(args.output)
        (args.output / "training.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "steps": args.steps,
                    "losses": losses,
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "elapsed_s": time.perf_counter() - start,
                    "seed": args.seed,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        model.remote_layers.sequence_manager.shutdown()


if __name__ == "__main__":
    main()
