#!/usr/bin/env python3
"""
Real Activation Dumper for BloomBee Distributed Inference

This script captures REAL intermediate layer activations during actual
model inference by hooking into the TransformerBackend.

Usage:
    1. On SERVER side (where you run the BloomBee server):
       Enable activation dumping before starting the server:
       
       export BLOOMBEE_DUMP_ACTIVATIONS=1
       export BLOOMBEE_ACTIVATION_DIR=/tmp/real_activations
       export BLOOMBEE_ACTIVATION_SAMPLES=20
       
       Then start your server normally.
       
    2. On CLIENT side:
       Run inference as usual - activations will be captured on the server.

    3. After inference:
       Copy the activation files from server and run compression benchmark:
       
       python benchmarks/compression/benchmark_compression.py --input_dir /tmp/real_activations
"""

import os
import json
import time
import torch
import threading
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationMetadata:
    """Metadata for a captured activation."""
    filename: str
    step: int
    layer_idx: int
    block_uid: str
    shape: List[int]
    dtype: str
    numel: int
    size_bytes: int
    mean: float
    std: float
    min_val: float
    max_val: float
    timestamp: str
    inference_prefix_length: int
    batch_size: int
    seq_len: int
    phase: str
    source: str = ""
    channel: str = ""
    direction: str = ""
    model: str = ""
    prompt_len: int = 0
    blocks: str = ""
    compute_dtype: str = ""
    schema_dtype: str = ""
    wire_dtype: str = ""
    tensor_name: str = "hidden_states"


class RealActivationDumper:
    """
    Singleton class to capture real activations from TransformerBackend.
    
    This is designed to be integrated into the BloomBee server code.
    Enable by setting environment variable BLOOMBEE_DUMP_ACTIVATIONS=1
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.capture_backend = os.environ.get("BLOOMBEE_DUMP_ACTIVATIONS", "0") == "1"
        self.capture_wire = os.environ.get("BLOOMBEE_DUMP_WIRE_ACTIVATIONS", "0") == "1"
        self.enabled = self.capture_backend or self.capture_wire
        self.output_dir = Path(os.environ.get("BLOOMBEE_ACTIVATION_DIR", "/tmp/real_activations"))
        self.max_samples = int(os.environ.get("BLOOMBEE_ACTIVATION_SAMPLES", "20"))
        raw_phases = os.environ.get("BLOOMBEE_ACTIVATION_PHASES", "prefill,decode")
        self.allowed_phases = {
            item.strip().lower()
            for item in raw_phases.split(",")
            if item.strip()
        }
        if not self.allowed_phases:
            self.allowed_phases = {"prefill", "decode"}
        
        self.saved_count = 0
        self.step_count = 0
        self.metadata_list: List[Dict] = []
        self._save_lock = threading.Lock()
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "[ACTIVATION_DUMPER] Enabled: "
                f"output_dir={self.output_dir}, "
                f"max_samples={self.max_samples}, "
                f"phases={sorted(self.allowed_phases)}"
            )
        else:
            logger.debug("[ACTIVATION_DUMPER] Disabled. Set BLOOMBEE_DUMP_ACTIVATIONS=1 to enable.")
    
    def should_capture(self, phase: str) -> bool:
        """Check if we should capture the current step."""
        return self.enabled and self.saved_count < self.max_samples and phase in self.allowed_phases

    @staticmethod
    def infer_phase(seq_len: int) -> str:
        # In current inference flow, seq_len==1 corresponds to decode and seq_len>1 to prefill.
        return "decode" if int(seq_len) == 1 else "prefill"
    
    def capture(
        self,
        hidden_states: torch.Tensor,
        block_uid: str = "unknown",
        layer_idx: int = 0,
        inference_info: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Capture a real activation tensor from inference.
        
        This should be called from TransformerBackend.inference_step()
        
        Args:
            hidden_states: The activation tensor [batch_size, seq_len, hidden_size]
            block_uid: The block identifier
            layer_idx: Layer index 
            inference_info: Optional InferenceMetadata for additional context
            
        Returns:
            Path to saved file, or None if not captured
        """
        context = dict(context or {})
        is_wire_capture = bool(context.pop("wire_capture", False))
        if is_wire_capture and not self.capture_wire:
            return None
        if not is_wire_capture and not self.capture_backend:
            return None

        seq_len = hidden_states.shape[1] if hidden_states.ndim >= 2 else 1
        phase = str(context.get("phase") or self.infer_phase(seq_len)).lower()
        if not self.should_capture(phase):
            return None
        
        with self._save_lock:
            if self.saved_count >= self.max_samples or phase not in self.allowed_phases:
                return None
                
            self.step_count += 1
            
            # Copy tensor to CPU (avoid modifying original)
            tensor = hidden_states.detach().cpu().clone()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"real_activation_layer{layer_idx}_step{self.step_count}_{timestamp}.pt"
            filepath = self.output_dir / filename
            
            # Save tensor
            torch.save(tensor, filepath)
            
            # Extract info from inference_info if available
            prefix_length = 0
            if inference_info is not None:
                prefix_length = getattr(inference_info, 'prefix_length', 0)
            prompt_len = context.get("prompt_len", prefix_length)
            try:
                prompt_len = int(prompt_len)
            except Exception:
                prompt_len = 0
            
            context_batch_size = context.get("batch_size")
            try:
                batch_size = int(context_batch_size) if context_batch_size is not None else (
                    tensor.shape[0] if tensor.ndim >= 1 else 1
                )
            except Exception:
                batch_size = tensor.shape[0] if tensor.ndim >= 1 else 1
            seq_len = tensor.shape[1] if tensor.ndim >= 2 else 1
            
            # Record metadata
            metadata = ActivationMetadata(
                filename=filename,
                step=self.step_count,
                layer_idx=layer_idx,
                block_uid=block_uid,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                numel=tensor.numel(),
                size_bytes=tensor.numel() * tensor.element_size(),
                mean=float(tensor.float().mean()),
                std=float(tensor.float().std()),
                min_val=float(tensor.float().min()),
                max_val=float(tensor.float().max()),
                timestamp=timestamp,
                inference_prefix_length=prefix_length,
                batch_size=batch_size,
                seq_len=seq_len,
                phase=phase,
                source=str(context.get("source", "")),
                channel=str(context.get("channel", "")),
                direction=str(context.get("direction", "")),
                model=str(context.get("model", "")),
                prompt_len=prompt_len,
                blocks=str(context.get("blocks", "")),
                compute_dtype=str(context.get("compute_dtype", "")),
                schema_dtype=str(context.get("schema_dtype", "")),
                wire_dtype=str(context.get("wire_dtype", str(tensor.dtype).replace("torch.", ""))),
                tensor_name=str(context.get("tensor_name", "hidden_states")),
            )
            
            self.metadata_list.append(asdict(metadata))
            self.saved_count += 1
            
            logger.info(
                f"[ACTIVATION_DUMPER] Captured: {filename} | "
                f"phase={phase} | "
                f"shape={list(tensor.shape)} | size={metadata.size_bytes/1024:.1f}KB | "
                f"mean={metadata.mean:.4f} | std={metadata.std:.4f}"
            )
            
            # Save metadata after each capture (in case of crash)
            self._save_metadata()
            
            return str(filepath)
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = self.output_dir / "metadata.json"
        lock_file = self.output_dir / "metadata.lock"
        current_samples = [dict(item) for item in self.metadata_list]

        with open(lock_file, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing_samples: List[Dict] = []
            if metadata_file.exists():
                try:
                    existing = json.loads(metadata_file.read_text())
                    if isinstance(existing, dict):
                        existing_samples = [
                            dict(item)
                            for item in existing.get("samples", [])
                            if isinstance(item, dict)
                        ]
                except Exception:
                    existing_samples = []

            merged_by_filename: Dict[str, Dict] = {}
            for item in existing_samples + current_samples:
                filename = str(item.get("filename", ""))
                if not filename:
                    continue
                merged_by_filename[filename] = item
            merged_samples = list(merged_by_filename.values())

            summary = {
                "total_samples": len(merged_samples),
                "total_steps": self.step_count,
                "max_samples": self.max_samples,
                "created_at": datetime.now().isoformat(),
                "samples": merged_samples,
            }

            with open(metadata_file, "w") as f:
                json.dump(summary, f, indent=2)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    def get_summary(self) -> Dict:
        """Get summary of captured activations."""
        if not self.metadata_list:
            return {"error": "No activations captured"}
        
        total_bytes = sum(m["size_bytes"] for m in self.metadata_list)
        
        return {
            "enabled": self.enabled,
            "num_samples": self.saved_count,
            "max_samples": self.max_samples,
            "total_size_mb": total_bytes / (1024 * 1024),
            "output_dir": str(self.output_dir),
        }
    
    @classmethod
    def get_instance(cls) -> "RealActivationDumper":
        """Get the singleton instance."""
        return cls()


# Global instance for easy access
_dumper = None


def get_activation_dumper() -> RealActivationDumper:
    """Get the global activation dumper instance."""
    global _dumper
    if _dumper is None:
        _dumper = RealActivationDumper()
    return _dumper


def capture_activation(
    hidden_states: torch.Tensor,
    block_uid: str = "unknown",
    layer_idx: int = 0,
    inference_info: Optional[Any] = None,
) -> Optional[str]:
    """
    Convenience function to capture activation from anywhere in the code.
    
    Usage in TransformerBackend.inference_step():
    
        from bloombee.utils.real_activation_dumper import capture_activation
        
        # At the start of inference_step, after receiving hidden_states:
        capture_activation(
            hidden_states,
            block_uid=self.name,
            layer_idx=0,  # or extract from self.name
            inference_info=inference_info
        )
    """
    dumper = get_activation_dumper()
    return dumper.capture(hidden_states, block_uid, layer_idx, inference_info)


def capture_wire_activation(
    hidden_states: torch.Tensor,
    *,
    source: str,
    channel: str,
    direction: str,
    phase: str,
    blocks: str = "",
    tensor_name: str = "hidden_states",
    compute_dtype: str = "",
    schema_dtype: str = "",
    wire_dtype: str = "",
    batch_size: Optional[int] = None,
    prompt_len: int = 0,
    model: str = "",
) -> Optional[str]:
    """
    Capture a tensor exactly at a wire serialization point for codec benchmarks.

    Disabled unless BLOOMBEE_DUMP_WIRE_ACTIVATIONS=1.
    """
    if os.environ.get("BLOOMBEE_DUMP_WIRE_ACTIVATIONS", "0") != "1":
        return None
    context = {
        "wire_capture": True,
        "source": source,
        "channel": channel,
        "direction": direction,
        "phase": phase,
        "blocks": blocks,
        "tensor_name": tensor_name,
        "compute_dtype": compute_dtype,
        "schema_dtype": schema_dtype,
        "wire_dtype": wire_dtype,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "model": model,
    }
    dumper = get_activation_dumper()
    return dumper.capture(hidden_states, block_uid=blocks or channel, layer_idx=0, context=context)


# ============================================================
# Integration Code for TransformerBackend
# ============================================================

INTEGRATION_CODE = '''
# Add this import at the top of backend.py:
from bloombee.utils.real_activation_dumper import capture_activation

# Add this line at the START of TransformerBackend.inference_step(), 
# right after the assert statement (around line 262):

        # [ACTIVATION_DUMP] Capture real hidden_states for compression analysis
        capture_activation(
            hidden_states,
            block_uid=self.name,
            layer_idx=0,
            inference_info=inference_info
        )
'''


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Activation Dumper for BloomBee")
    parser.add_argument("--show-integration", action="store_true", 
                        help="Show integration code for backend.py")
    parser.add_argument("--test", action="store_true",
                        help="Test the dumper with synthetic data")
    
    args = parser.parse_args()
    
    if args.show_integration:
        print("=" * 60)
        print("Integration Instructions")
        print("=" * 60)
        print(INTEGRATION_CODE)
        print("\n" + "=" * 60)
        print("Environment Variables:")
        print("=" * 60)
        print("""
export BLOOMBEE_DUMP_ACTIVATIONS=1
export BLOOMBEE_ACTIVATION_DIR=/tmp/real_activations
export BLOOMBEE_ACTIVATION_SAMPLES=20
export BLOOMBEE_ACTIVATION_PHASES=decode
""")
    elif args.test:
        os.environ["BLOOMBEE_DUMP_ACTIVATIONS"] = "1"
        os.environ["BLOOMBEE_ACTIVATION_DIR"] = "/tmp/test_real_activations"
        os.environ["BLOOMBEE_ACTIVATION_SAMPLES"] = "5"
        
        dumper = RealActivationDumper()
        
        print("Testing Real Activation Dumper...")
        for i in range(5):
            tensor = torch.randn(1, 512, 4096, dtype=torch.float16)
            result = dumper.capture(tensor, block_uid=f"block_{i}", layer_idx=i)
            print(f"  Captured: {result}")
        
        print("\nSummary:", dumper.get_summary())
    else:
        parser.print_help()
