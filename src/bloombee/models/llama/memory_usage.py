import torch
from pynvml.smi import nvidia_smi
from pynvml import *
from typing import Optional

def nvidia_smi_usage():
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	return (info.used) / 1024 / 1024 / 1024

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

# input("Press Any Key To Continue ..")
