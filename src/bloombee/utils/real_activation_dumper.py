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
       
       python benchmarks/benchmark_compression.py --input_dir /tmp/real_activations
"""

import os
import json
import time
import torch
import threading
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
        self.enabled = os.environ.get("BLOOMBEE_DUMP_ACTIVATIONS", "0") == "1"
        self.output_dir = Path(os.environ.get("BLOOMBEE_ACTIVATION_DIR", "/tmp/real_activations"))
        self.max_samples = int(os.environ.get("BLOOMBEE_ACTIVATION_SAMPLES", "20"))
        
        self.saved_count = 0
        self.step_count = 0
        self.metadata_list: List[Dict] = []
        self._save_lock = threading.Lock()
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ACTIVATION_DUMPER] Enabled: output_dir={self.output_dir}, max_samples={self.max_samples}")
        else:
            logger.debug("[ACTIVATION_DUMPER] Disabled. Set BLOOMBEE_DUMP_ACTIVATIONS=1 to enable.")
    
    def should_capture(self) -> bool:
        """Check if we should capture the current step."""
        return self.enabled and self.saved_count < self.max_samples
    
    def capture(
        self,
        hidden_states: torch.Tensor,
        block_uid: str = "unknown",
        layer_idx: int = 0,
        inference_info: Optional[Any] = None,
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
        if not self.should_capture():
            return None
        
        with self._save_lock:
            if self.saved_count >= self.max_samples:
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
            )
            
            self.metadata_list.append(asdict(metadata))
            self.saved_count += 1
            
            logger.info(
                f"[ACTIVATION_DUMPER] Captured: {filename} | "
                f"shape={list(tensor.shape)} | size={metadata.size_bytes/1024:.1f}KB | "
                f"mean={metadata.mean:.4f} | std={metadata.std:.4f}"
            )
            
            # Save metadata after each capture (in case of crash)
            self._save_metadata()
            
            return str(filepath)
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = self.output_dir / "metadata.json"
        
        summary = {
            "total_samples": self.saved_count,
            "total_steps": self.step_count,
            "max_samples": self.max_samples,
            "created_at": datetime.now().isoformat(),
            "samples": self.metadata_list,
        }
        
        with open(metadata_file, "w") as f:
            json.dump(summary, f, indent=2)
    
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
