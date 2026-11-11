"""Export a fixed adapter as FP16 HF weights for BloomBee's ordinary KV path.

Merge on CPU in FP32, then cast once. Requires RAM for the FP32 base model.
Deploy under a new model directory and DHT prefix; never overwrite a live base.
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Merged output must be a new directory")
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type not in ("llama", "qwen3"):
        raise ValueError("This example supports the dense LLaMA/Qwen3 LoRA training path")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload(safe_merge=True)
    model.half().save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"Merged FP16 model saved to {args.output}; use a new DHT prefix", flush=True)


if __name__ == "__main__":
    main()
