import dataclasses
import logging
from bloombee.flexgen_utils.compression import CompressionConfig

# Create logger for policy debugging
policy_logger = logging.getLogger('bloombee.policy')
policy_logger.setLevel(logging.INFO)

@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent
    
    def log_batch_size_strategy(self):
        """Log batch size strategy and resource allocation for debugging"""
        policy_logger.info(f"[POLICY_BATCH_SIZE] GPU Batch Size: {self.gpu_batch_size}")
        policy_logger.info(f"[POLICY_BATCH_SIZE] Num GPU Batches: {self.num_gpu_batches}")
        policy_logger.info(f"[POLICY_BATCH_SIZE] Total GPU Capacity: {self.gpu_batch_size * self.num_gpu_batches}")
        policy_logger.info(f"[POLICY_RESOURCES] Weight GPU%: {self.w_gpu_percent}% | CPU%: {self.w_cpu_percent}% | Disk%: {self.w_disk_percent}%")
        policy_logger.info(f"[POLICY_RESOURCES] Cache GPU%: {self.cache_gpu_percent}% | CPU%: {self.cache_cpu_percent}% | Disk%: {self.cache_disk_percent}%")
        policy_logger.info(f"[POLICY_RESOURCES] Activation GPU%: {self.act_gpu_percent}% | CPU%: {self.act_cpu_percent}% | Disk%: {self.act_disk_percent}%")
        policy_logger.info(f"[POLICY_OPTIMIZATION] Overlap: {self.overlap} | Sep Layer: {self.sep_layer} | Pin Weight: {self.pin_weight}")
        policy_logger.info(f"[POLICY_OPTIMIZATION] CPU Cache Compute: {self.cpu_cache_compute} | Attn Sparsity: {self.attn_sparsity}")
        policy_logger.info(f"[POLICY_COMPRESSION] Compress Weight: {self.compress_weight} | Compress Cache: {self.compress_cache}")

