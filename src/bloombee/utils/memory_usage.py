import os
import torch
# from pynvml.smi import nvidia_smi
from pynvml import *
from typing import Optional
import logging

# [MBPIPE_DEBUG] Create a dedicated logger for micro-batch memory debugging
_mbpipe_mem_logger = logging.getLogger('bloombee.mbpipe_memory')
_mbpipe_mem_logger.setLevel(logging.INFO)

def nvidia_smi_usage():
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	return (info.used) / 1024 / 1024 / 1024


# =============================================================================
# [MBPIPE_DEBUG] Micro-batch Memory Debugging Utilities
# =============================================================================

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_gpu_memory_detailed() -> dict:
    """Get detailed GPU memory information."""
    result = {
        'allocated_mb': 0.0,
        'reserved_mb': 0.0,
        'max_allocated_mb': 0.0,
        'nvidia_smi_mb': 0.0,
    }
    
    if torch.cuda.is_available():
        result['allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        result['reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        result['max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            result['nvidia_smi_mb'] = info.used / (1024 * 1024)
        except Exception:
            pass
    
    return result


def log_mbpipe_memory(tag: str, extra_info: str = ""):
    """
    Log GPU memory usage with [MBPIPE_MEM] prefix for easy filtering.
    
    Usage:
        log_mbpipe_memory("before_kv_alloc", "batch_size=8")
        log_mbpipe_memory("after_mb_compute", "mb_idx=0")
    """
    mem = get_gpu_memory_detailed()
    extra = f" | {extra_info}" if extra_info else ""
    
    _mbpipe_mem_logger.info(
        f"[MBPIPE_MEM] {tag}: "
        f"alloc={mem['allocated_mb']:.1f}MB, "
        f"reserved={mem['reserved_mb']:.1f}MB, "
        f"max_alloc={mem['max_allocated_mb']:.1f}MB, "
        f"nvidia_smi={mem['nvidia_smi_mb']:.1f}MB{extra}"
    )


def log_kv_cache_allocation(
    batch_size: int,
    micro_batch_size: int,
    max_length: int,
    num_blocks: int,
    hidden_size: int,
    num_heads: int,
    dtype_bytes: int = 2,  # fp16 = 2 bytes
):
    """
    Log expected vs actual KV cache allocation.
    
    This calculates:
    - Expected KV cache size for FULL batch
    - Expected KV cache size for micro-batch only
    - Expected memory savings if we used micro-batch-sized allocation
    """
    # KV cache shape: (S, B*H, D) for K and V
    # where S = max_length, B = batch_size, H = num_heads, D = hidden_size/num_heads
    head_dim = hidden_size // num_heads
    
    # Full batch KV cache size (current implementation)
    full_kv_size_per_block = 2 * max_length * batch_size * num_heads * head_dim * dtype_bytes
    full_kv_total_mb = (full_kv_size_per_block * num_blocks) / (1024 * 1024)
    
    # Micro-batch KV cache size (optimal implementation)
    mb_kv_size_per_block = 2 * max_length * micro_batch_size * num_heads * head_dim * dtype_bytes
    mb_kv_total_mb = (mb_kv_size_per_block * num_blocks) / (1024 * 1024)
    
    # Memory savings
    savings_mb = full_kv_total_mb - mb_kv_total_mb
    savings_ratio = (savings_mb / full_kv_total_mb * 100) if full_kv_total_mb > 0 else 0
    
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] ==================== KV CACHE ALLOCATION DEBUG ===================="
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] Input params: batch_size={batch_size}, micro_batch_size={micro_batch_size}, "
        f"max_length={max_length}, num_blocks={num_blocks}"
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] Model params: hidden_size={hidden_size}, num_heads={num_heads}, "
        f"head_dim={head_dim}, dtype_bytes={dtype_bytes}"
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] CURRENT (full batch): {full_kv_total_mb:.2f} MB for {num_blocks} blocks"
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] OPTIMAL (micro-batch): {mb_kv_total_mb:.2f} MB for {num_blocks} blocks"
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] POTENTIAL SAVINGS: {savings_mb:.2f} MB ({savings_ratio:.1f}%)"
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] NOTE: If alloc_batch_size == batch_size, NO memory savings occur!"
    )
    _mbpipe_mem_logger.info(
        f"[MBPIPE_KV_ALLOC] =================================================================="
    )


def log_kv_cache_shape(
    cache_name: str,
    shape: tuple,
    batch_offset: int = 0,
    full_batch_size: int = 0,
    micro_batch_size: int = 0,
):
    """
    Log actual KV cache tensor shape for debugging.
    """
    if len(shape) == 3:
        S, BH, D = shape
        _mbpipe_mem_logger.info(
            f"[MBPIPE_KV_SHAPE] {cache_name}: shape=(S={S}, BH={BH}, D={D}), "
            f"batch_offset={batch_offset}, full_batch_size={full_batch_size}, "
            f"micro_batch_size={micro_batch_size}"
        )
    else:
        _mbpipe_mem_logger.info(
            f"[MBPIPE_KV_SHAPE] {cache_name}: shape={shape}, "
            f"batch_offset={batch_offset}, full_batch_size={full_batch_size}, "
            f"micro_batch_size={micro_batch_size}"
        )


class MemoryTracker:
    """
    Context manager to track memory changes during a code block.
    
    Usage:
        with MemoryTracker("kv_cache_allocation") as tracker:
            # ... allocation code ...
        # Logs memory delta when exiting
    """
    
    def __init__(self, tag: str, extra_info: str = ""):
        self.tag = tag
        self.extra_info = extra_info
        self.start_mem = None
        self.end_mem = None
    
    def __enter__(self):
        self.start_mem = get_gpu_memory_detailed()
        log_mbpipe_memory(f"{self.tag}_START", self.extra_info)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_mem = get_gpu_memory_detailed()
        
        delta = self.end_mem['allocated_mb'] - self.start_mem['allocated_mb']
        delta_reserved = self.end_mem['reserved_mb'] - self.start_mem['reserved_mb']
        
        _mbpipe_mem_logger.info(
            f"[MBPIPE_MEM] {self.tag}_END: "
            f"delta_alloc={delta:+.1f}MB, delta_reserved={delta_reserved:+.1f}MB | {self.extra_info}"
        )
        
        return False  # Don't suppress exceptions
    
    @property
    def delta_allocated_mb(self) -> float:
        if self.start_mem and self.end_mem:
            return self.end_mem['allocated_mb'] - self.start_mem['allocated_mb']
        return 0.0

def get_memory_stats() -> dict:
	"""Get detailed memory statistics from both PyTorch and nvidia-smi."""
	stats = {}
	
	# Get nvidia-smi memory usage
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	stats['nvidia_smi_used'] = info.used / 1024 / 1024 / 1024  # GB
	
	# Get PyTorch memory stats
	stats['torch_allocated'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
	stats['torch_max_allocated'] = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB
	
	return stats

def see_memory_usage(message: str, force: bool = True):
	"""Print current memory usage with a message."""
	stats = get_memory_stats()
	logger = f"{message}\n"
	logger += f"Nvidia-smi: {stats['nvidia_smi_used']:.2f} GB\n"
	logger += f"Memory Allocated: {stats['torch_allocated']:.2f} GB\n"
	logger += f"Max Memory Allocated: {stats['torch_max_allocated']:.2f} GB\n"
	print(logger)

def memlog_enabled() -> bool:
	"""Return True if memory logging is enabled via env var BB_MEMLOG."""
	val = os.environ.get("BB_MEMLOG", "0")
	return str(val).lower() in ("1", "true", "yes", "on")

def log_mem(message: str):
    """Conditionally log memory usage when BB_MEMLOG is enabled."""
    # if memlog_enabled():
    #     see_memory_usage(message)
    pass

def profile_weight_init(func):
	"""Decorator to profile memory usage during weight initialization."""
	def wrapper(*args, **kwargs):
		# Record initial memory usage
		initial_stats = get_memory_stats()
		initial_memory = initial_stats['nvidia_smi_used']
		
		# Execute the function
		result = func(*args, **kwargs)
		
		# Record final memory usage
		final_stats = get_memory_stats()
		final_memory = final_stats['nvidia_smi_used']
		
		# Calculate and print memory usage
		memory_used = final_memory - initial_memory
		print(f"\nWeight Initialization Memory Profile:")
		print(f"Initial Memory: {initial_memory:.2f} GB")
		print(f"Final Memory: {final_memory:.2f} GB")
		print(f"Memory Used: {memory_used:.2f} GB")
		print(f"PyTorch Allocated: {final_stats['torch_allocated']:.2f} GB")
		print(f"PyTorch Max Allocated: {final_stats['torch_max_allocated']:.2f} GB")
		
		return result
	return wrapper 