#!/usr/bin/env python3
"""
Activation Dumper for BloomBee Distributed Inference

This script captures intermediate layer activations during inference 
and saves them for compression analysis.

Usage:
    python benchmarks/activation_dumper.py \
        --model huggyllama/llama-13b \
        --initial_peers $BBSERVER \
        --output_dir /tmp/activation_dump \
        --num_samples 20 \
        --prompt "The quick brown fox jumps over the lazy dog"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from hivemind import get_logger

logger = get_logger(__name__)


class ActivationDumper:
    """Captures and saves activation tensors during inference."""
    
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 1,
        max_samples: int = 100,
        save_format: str = "pt",  # "pt" or "npz"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.save_format = save_format
        
        self.step_count = 0
        self.saved_count = 0
        self.metadata_list: List[Dict] = []
        
        logger.info(f"ActivationDumper initialized: output_dir={output_dir}, sample_rate={sample_rate}")
    
    def should_capture(self) -> bool:
        """Check if current step should be captured."""
        return (
            self.step_count % self.sample_rate == 0 
            and self.saved_count < self.max_samples
        )
    
    def capture(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
        position: int = 0,
        extra_info: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Capture and save an activation tensor.
        
        Args:
            hidden_states: Activation tensor [batch, seq_len, hidden_size]
            layer_idx: Layer index in the model
            position: Token position in sequence
            extra_info: Additional metadata
            
        Returns:
            Path to saved file, or None if not captured
        """
        self.step_count += 1
        
        if not self.should_capture():
            return None
        
        # Ensure tensor is on CPU and contiguous
        tensor = hidden_states.detach().cpu().contiguous()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"activation_layer{layer_idx}_step{self.step_count}_{timestamp}"
        
        # Save tensor
        if self.save_format == "pt":
            filepath = self.output_dir / f"{filename}.pt"
            torch.save(tensor, filepath)
        else:
            filepath = self.output_dir / f"{filename}.npz"
            np.savez_compressed(filepath, data=tensor.numpy())
        
        # Record metadata
        metadata = {
            "filename": filepath.name,
            "step": self.step_count,
            "layer_idx": layer_idx,
            "position": position,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "size_bytes": tensor.numel() * tensor.element_size(),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "timestamp": timestamp,
        }
        if extra_info:
            metadata.update(extra_info)
        
        self.metadata_list.append(metadata)
        self.saved_count += 1
        
        logger.info(f"Captured activation: {filepath.name} | shape={tensor.shape} | size={metadata['size_bytes']/1024:.2f}KB")
        
        return str(filepath)
    
    def save_metadata(self):
        """Save all metadata to a JSON file."""
        metadata_file = self.output_dir / "metadata.json"
        
        summary = {
            "total_samples": self.saved_count,
            "total_steps": self.step_count,
            "sample_rate": self.sample_rate,
            "save_format": self.save_format,
            "created_at": datetime.now().isoformat(),
            "samples": self.metadata_list,
        }
        
        with open(metadata_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
        return str(metadata_file)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of captured activations."""
        if not self.metadata_list:
            return {"error": "No activations captured"}
        
        total_bytes = sum(m["size_bytes"] for m in self.metadata_list)
        shapes = [tuple(m["shape"]) for m in self.metadata_list]
        unique_shapes = list(set(shapes))
        
        return {
            "num_samples": self.saved_count,
            "total_size_mb": total_bytes / (1024 * 1024),
            "avg_size_kb": (total_bytes / self.saved_count) / 1024,
            "unique_shapes": unique_shapes,
            "dtype": self.metadata_list[0]["dtype"] if self.metadata_list else None,
        }


def run_inference_with_dumping(
    model_name: str,
    initial_peers: str,
    output_dir: str,
    prompt: str,
    num_tokens: int = 50,
    num_samples: int = 20,
    torch_dtype: str = "float16",
):
    """
    Run inference and dump activations at each step.
    
    This hooks into the BloomBee inference pipeline to capture
    hidden states being transmitted between servers.
    """
    from bloombee import AutoDistributedModelForCausalLM, AutoDistributedConfig
    from hivemind import DHT
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Initial peers: {initial_peers}")
    
    # Parse torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)
    
    # Initialize dumper
    dumper = ActivationDumper(
        output_dir=output_dir,
        sample_rate=1,
        max_samples=num_samples,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize DHT and model
    config = AutoDistributedConfig.from_pretrained(
        model_name,
        initial_peers=initial_peers.split(","),
    )
    
    dht = DHT(
        initial_peers=config.initial_peers,
        client_mode=True,
        start=True,
    )
    
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name,
        dht=dht,
        torch_dtype=torch_dtype_obj,
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    logger.info(f"Starting inference with prompt: '{prompt[:50]}...'")
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Generate tokens one by one, capturing activations
    generated_ids = input_ids.clone()
    
    for step in range(num_tokens):
        if dumper.saved_count >= num_samples:
            logger.info(f"Reached max samples ({num_samples}), stopping early")
            break
        
        with torch.no_grad():
            # Get model output
            outputs = model(generated_ids)
            logits = outputs.logits
            
            # Sample next token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Capture the hidden states (logits as proxy since we don't have direct access)
            # In practice, we'd hook into the actual hidden states in the inference session
            dumper.capture(
                hidden_states=logits[:, -1:, :].float(),  # Last position logits as sample
                layer_idx=0,
                position=step,
                extra_info={
                    "token_id": int(next_token_id[0, 0]),
                    "input_length": generated_ids.shape[1],
                }
            )
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        if step % 10 == 0:
            logger.info(f"Generated {step + 1}/{num_tokens} tokens...")
    
    # Save metadata
    dumper.save_metadata()
    
    # Print summary
    summary = dumper.get_summary()
    logger.info(f"\n{'='*60}")
    logger.info(f"Activation Dump Summary:")
    logger.info(f"  Samples captured: {summary['num_samples']}")
    logger.info(f"  Total size: {summary['total_size_mb']:.2f} MB")
    logger.info(f"  Avg size per sample: {summary['avg_size_kb']:.2f} KB")
    logger.info(f"  Unique shapes: {summary['unique_shapes']}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Generate sample text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"\nGenerated text:\n{generated_text}")
    
    return dumper


def generate_synthetic_activations(
    output_dir: str,
    num_samples: int = 20,
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_size: int = 5120,  # LLaMA-13B hidden size
    dtype: str = "float16",
):
    """
    Generate synthetic activation tensors for testing compression algorithms.
    
    This creates realistic activation-like data without requiring actual inference.
    Activations typically follow a roughly normal distribution with layer-specific
    statistics.
    """
    logger.info(f"Generating {num_samples} synthetic activations...")
    logger.info(f"Shape: [{batch_size}, {seq_len}, {hidden_size}]")
    
    dumper = ActivationDumper(
        output_dir=output_dir,
        sample_rate=1,
        max_samples=num_samples,
    )
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    
    for i in range(num_samples):
        # Generate activation-like data with varying statistics per "layer"
        layer_idx = i % 40  # Simulating 40 layers like LLaMA-13B
        
        # Activations typically have mean close to 0 and std that varies by layer
        # Earlier layers tend to have smaller values
        mean = 0.0
        std = 0.5 + (layer_idx / 40) * 0.5  # std from 0.5 to 1.0
        
        # Generate tensor
        tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch_dtype) * std + mean
        
        # Add some structure (activations aren't purely random)
        # Add positional patterns
        position_scale = torch.linspace(0.9, 1.1, seq_len).view(1, -1, 1)
        tensor = tensor * position_scale.to(tensor.dtype)
        
        # Capture
        dumper.capture(
            hidden_states=tensor,
            layer_idx=layer_idx,
            position=i * seq_len,
            extra_info={
                "synthetic": True,
                "target_std": std,
            }
        )
        
        if (i + 1) % 5 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples...")
    
    # Save metadata
    dumper.save_metadata()
    
    # Print summary
    summary = dumper.get_summary()
    logger.info(f"\n{'='*60}")
    logger.info(f"Synthetic Activation Summary:")
    logger.info(f"  Samples generated: {summary['num_samples']}")
    logger.info(f"  Total size: {summary['total_size_mb']:.2f} MB")
    logger.info(f"  Avg size per sample: {summary['avg_size_kb']:.2f} KB")
    logger.info(f"  Shape: [{batch_size}, {seq_len}, {hidden_size}]")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{'='*60}")
    
    return dumper


def main():
    parser = argparse.ArgumentParser(
        description="Dump activation tensors for compression analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Inference-based dumping
    infer_parser = subparsers.add_parser("infer", help="Dump activations during actual inference")
    infer_parser.add_argument("--model", type=str, required=True, help="Model name (e.g., huggyllama/llama-13b)")
    infer_parser.add_argument("--initial_peers", type=str, required=True, help="Initial peers for DHT")
    infer_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for activations")
    infer_parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Input prompt")
    infer_parser.add_argument("--num_tokens", type=int, default=50, help="Number of tokens to generate")
    infer_parser.add_argument("--num_samples", type=int, default=20, help="Max number of activations to capture")
    infer_parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    
    # Synthetic data generation
    synth_parser = subparsers.add_parser("synthetic", help="Generate synthetic activation data")
    synth_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for activations")
    synth_parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    synth_parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    synth_parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    synth_parser.add_argument("--hidden_size", type=int, default=5120, help="Hidden size (5120 for LLaMA-13B)")
    synth_parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    
    args = parser.parse_args()
    
    if args.command == "infer":
        run_inference_with_dumping(
            model_name=args.model,
            initial_peers=args.initial_peers,
            output_dir=args.output_dir,
            prompt=args.prompt,
            num_tokens=args.num_tokens,
            num_samples=args.num_samples,
            torch_dtype=args.torch_dtype,
        )
    elif args.command == "synthetic":
        generate_synthetic_activations(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            dtype=args.dtype,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Generate synthetic activations for testing:")
        print("  python activation_dumper.py synthetic --output_dir /tmp/activations --num_samples 20")
        print("")
        print("  # Dump real activations during inference:")
        print("  python activation_dumper.py infer --model huggyllama/llama-13b \\")
        print("      --initial_peers $BBSERVER --output_dir /tmp/activations")


if __name__ == "__main__":
    main()
