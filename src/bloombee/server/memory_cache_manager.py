"""
Memory cache manager for KV cache offloading
Supports unified cache interface and device synchronization
"""

import contextlib
import logging
import multiprocessing as mp
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from bloombee.data_structures import KVCache, KVCacheMetadata, Handle
from bloombee.flexgen_utils.policy import Policy
from bloombee.server.memory_cache import MemoryCache, UnifiedCache, DeviceInfo

# Create dedicated offloading debug logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


def init_cache_manager_shared(policy, layer_id):
    """
    Shared function to initialize KVCacheManager
    Used by both flex_llama.py and block.py to eliminate code duplication
    
    Args:
        policy: Policy object containing cache configuration
        layer_id: Layer ID for logging purposes
        
    Returns:
        KVCacheManager instance or None if initialization fails
    """
    try:
        # Get cache size from policy, use default if not available
        cache_size = getattr(policy, 'cache_size', 1024 * 1024 * 1024)  # 1GB default
        max_alloc_timeout = getattr(policy, 'max_alloc_timeout', 30)
        
        cache_manager = KVCacheManager(cache_size, max_alloc_timeout, policy)
        # offload_logger.info(f" Initialized KVCacheManager - layer:{layer_id}")
        
        return cache_manager
        
    except ImportError as e:
        print(f"Warning: Could not import KVCacheManager: {e}")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize KVCacheManager: {e}")
        return None


class KVCacheManager:
    """
    Basic KV cache manager supporting offloading functionality
    Supports layered optimization by model layer and GPU batch
    Supports unified UnifiedCache interface
    """
    
    # Singleton pattern implementation
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, cache_size: int, max_alloc_timeout: int, policy: Policy):
        # Avoid duplicate initialization
        if self._initialized:
            return
        self._initialized = True
        
        self.cache_size = cache_size
        self.max_alloc_timeout = max_alloc_timeout
        self.policy = policy
        
        # Initialize memory cache
        self.cache = MemoryCache(cache_size, max_alloc_timeout)
        
        # Initialize device allocation strategy
        self.device_allocation = self._init_device_allocation()
        
        # Cache hierarchy: cache_hierarchy[layer_id][batch_id][position] = UnifiedCache handle
        self.cache_hierarchy: Dict[int, Dict[int, Dict[int, Handle]]] = {}
        
        # Statistics tracking
        self.stats = {
            'total_stores': 0,
            'total_loads': 0,
            'total_bytes_stored': 0,
            'total_bytes_loaded': 0,
            'device_transfers': 0
        }
        
        offload_logger.info("Initializing KVCacheManager - starting offloading system")
        offload_logger.info(f"Cache size: {cache_size / (1024*1024*1024):.2f} GB")
        offload_logger.info(f"Max alloc timeout: {max_alloc_timeout} seconds")
        offload_logger.info(f"Policy: GPU={self.policy.cache_gpu_percent}%, CPU={self.policy.cache_cpu_percent}%, Disk={self.policy.cache_disk_percent}%")
        offload_logger.info("KVCacheManager initialization completed")

    def _init_device_allocation(self) -> Dict[str, float]:
        """Initialize device allocation strategy based on policy"""
        return {
            'gpu': self.policy.cache_gpu_percent / 100.0,
            'cpu': self.policy.cache_cpu_percent / 100.0,
            'disk': self.policy.cache_disk_percent / 100.0
        }

    def clear(self):
        """Clear all cached data"""
        self.cache_hierarchy.clear()
        self.stats = {
            'total_stores': 0,
            'total_loads': 0,
            'total_bytes_stored': 0,
            'total_bytes_loaded': 0,
            'device_transfers': 0
        }
        offload_logger.info("KVCacheManager cache cleared")

    def init_cache_one_gpu_batch(self, layer_id: int, batch_id: int, 
                                config, task, policy) -> UnifiedCache:
        """
        Initialize cache for one GPU batch
        Creates UnifiedCache with appropriate device allocation
        """
        # Determine target device based on policy
        if self.policy.cache_gpu_percent == 100:
            device_type = 'gpu'
            device_id = 'cuda:0'
        elif self.policy.cache_cpu_percent == 100:
            device_type = 'cpu'
            device_id = 'cpu'
        elif self.policy.cache_disk_percent == 100:
            device_type = 'disk'
            device_id = '/tmp/disk_cache'
        else:
            # Mixed allocation - use GPU as primary
            device_type = 'gpu'
            device_id = 'cuda:0'
        
        # Create device info
        device_info = DeviceInfo(
            device_type=device_type,
            device_id=device_id,
            compression_config=self.policy.comp_cache_config if self.policy.compress_cache else None,
            offloaded=(device_type != 'gpu')
        )
        
        # Create empty UnifiedCache
        unified_cache = UnifiedCache(
            past_key_value=None,  # Will be populated during inference
            device_info=device_info
        )
        
        offload_logger.info(f"Initialized cache for layer {layer_id}, batch {batch_id}")
        offload_logger.info(f"Device: {device_type} ({device_id})")
        offload_logger.info(f"Compression: {self.policy.compress_cache}")
        
        return unified_cache

    def load_cache(self, position: int, layer_id: int, batch_id: int, 
                  target_device: str = 'cuda:0') -> Optional[UnifiedCache]:
        """
        Load cache from storage
        Supports position mapping/fallback mechanism
        """
        # First try to load from requested position
        handle = self._get_cache_handle(position, layer_id, batch_id)
        
        # If not found, try position mapping (cache usually stored at position 0 during inference)
        if handle is None:
            offload_logger.info(f"Position {position} not found, attempting position mapping...")
            
            # Try to find the latest available position
            if position > 0:
                # Find all available positions for this layer and batch
                available_positions = []
                if (layer_id in self.cache_hierarchy and 
                    batch_id in self.cache_hierarchy[layer_id]):
                    available_positions = list(self.cache_hierarchy[layer_id][batch_id].keys())
                
                if available_positions:
                    # Select the latest available position
                    available_positions.sort()
                    fallback_position = available_positions[-1]  # Select largest position (latest)
                    offload_logger.info(f"Trying to load from position {fallback_position} (available positions: {available_positions})")
                    handle = self._get_cache_handle(fallback_position, layer_id, batch_id)
                    
                    if handle is not None:
                        offload_logger.info(f"Position mapping successful: {position} -> {fallback_position}")
                    else:
                        offload_logger.info(f"Position {fallback_position} also not found")
                else:
                    offload_logger.info("No available cache positions")
        
        if handle is None:
            offload_logger.warning(f"Cache handle not found - position:{position}, layer:{layer_id}, batch:{batch_id}")
            offload_logger.warning("Position mapping failed")
            return None
        
        # Get UnifiedCache from MemoryCache using new storage key format
        storage_key = f"layer_{layer_id}_handle_{handle}"
        unified_cache = self.cache._unified_caches.get(storage_key)
        if unified_cache is None:
            offload_logger.warning(f"UnifiedCache not found - storage key:{storage_key}")
            offload_logger.warning(f"Available storage keys: {list(self.cache._unified_caches.keys())}")
            return None
        
        # Sync device if needed
        if unified_cache.device_info.device_id != target_device:
            offload_logger.info(f"Syncing cache from {unified_cache.device_info.device_id} to {target_device}")
            unified_cache = self.cache.sync_device_cache(unified_cache, target_device)
            self.stats['device_transfers'] += 1
        
        # Update statistics
        cache_size = self._calculate_cache_size(unified_cache)
        self.stats['total_loads'] += 1
        self.stats['total_bytes_loaded'] += cache_size
        
        offload_logger.info(f"Successfully loaded cache - position:{position}, layer:{layer_id}")
        offload_logger.info(f"Cache size: {cache_size / 1024:.1f} KB")
        offload_logger.info(f"Device: {unified_cache.device_info.device_id}")
        
        return unified_cache

    def store_cache(self, unified_cache: UnifiedCache, position: int, 
                   layer_id: int, batch_id: int) -> Handle:
        """
        Store cache to memory
        Generates unique handles per layer within inference step
        """
        if unified_cache.past_key_value is None:
            offload_logger.warning("Cannot store cache with None past_key_value")
            return -1
        
        # Generate global unique handle using class-level counter combined with layer_id
        if not hasattr(KVCacheManager, '_global_handle_counter'):
            KVCacheManager._global_handle_counter = 0
        
        base_handle = KVCacheManager._global_handle_counter
        handle = base_handle * 1000 + layer_id  # Reserve 1000 handles per layer
        KVCacheManager._global_handle_counter += 1
        
        offload_logger.info(f"Generated global handle: {handle} (base handle: {base_handle}, layer ID: {layer_id}, global counter: {KVCacheManager._global_handle_counter})")
        
        # Store UnifiedCache directly in MemoryCache using new storage key format
        storage_key = f"layer_{layer_id}_handle_{handle}"
        self.cache._unified_caches[storage_key] = unified_cache
        
        # Update hierarchy structure - use new handle format
        if layer_id not in self.cache_hierarchy:
            self.cache_hierarchy[layer_id] = {}
        if batch_id not in self.cache_hierarchy[layer_id]:
            self.cache_hierarchy[layer_id][batch_id] = {}
        
        # Store handle - use new handle format
        self.cache_hierarchy[layer_id][batch_id][position] = handle
        
        # Update statistics
        cache_size = self._calculate_cache_size(unified_cache)
        self.stats['total_stores'] += 1
        self.stats['total_bytes_stored'] += cache_size
        
        offload_logger.info(f"Successfully stored cache - position:{position}, layer:{layer_id}")
        offload_logger.info(f"Handle: {handle}")
        offload_logger.info(f"Device: {unified_cache.device_info}")
        offload_logger.info(f"Cache size: {cache_size / 1024:.1f} KB")
        offload_logger.info(f"Total memory used: {self.cache.current_size_bytes / (1024*1024):.1f} MB")
        offload_logger.info(f"Remaining memory: {self.cache.bytes_left / (1024*1024*1024):.1f} GB")
        
        return handle

    def add_cache(self, kvs: KVCache, start_position: int, layer_id: int = 0, batch_id: int = 0):
        """Add new cache data"""
        # Convert KVCache to UnifiedCache
        device_info = DeviceInfo(
            device_type='gpu',
            device_id='cuda:0',
            compression_config=self.policy.comp_cache_config if self.policy.compress_cache else None,
            offloaded=False
        )
        
        unified_cache = UnifiedCache(
            past_key_value=kvs.kvs,
            device_info=device_info
        )
        
        # Store using existing method
        handle = self.store_cache(unified_cache, start_position, layer_id, batch_id)
        offload_logger.info(f"Added cache with handle {handle}")

    def update_cache(self, new_kvs: KVCache, start_position: int, layer_id: int = 0, batch_id: int = 0):
        """Update existing cache data"""
        # Get existing cache
        existing_cache = self.load_cache(start_position, layer_id, batch_id)
        
        if existing_cache is None:
            # If no existing cache, add new one
            self.add_cache(new_kvs, start_position, layer_id, batch_id)
            return
        
        # Merge new KV tensors with existing ones
        if existing_cache.past_key_value and new_kvs.kvs:
            merged_kvs = []
            for existing, new in zip(existing_cache.past_key_value, new_kvs.kvs):
                if isinstance(existing, torch.Tensor) and isinstance(new, torch.Tensor):
                    merged = torch.cat([existing, new], dim=0)
                    merged_kvs.append(merged)
                else:
                    merged_kvs.append(new)
            
            # Create updated UnifiedCache
            updated_cache = UnifiedCache(
                past_key_value=tuple(merged_kvs),
                device_info=existing_cache.device_info,
                cache_handles=existing_cache.cache_handles
            )
            
            # Store updated cache
            handle = self.store_cache(updated_cache, start_position, layer_id, batch_id)
            offload_logger.info(f"Updated cache with handle {handle}")

    def bytes_left(self) -> int:
        """Get remaining cache capacity in bytes"""
        return self.cache.bytes_left

    def select_cache(self, kv_cache_position_ids: Optional[torch.Tensor] = None, 
                    layer_id: int = 0, batch_id: int = 0):
        """Select cache handles for given positions"""
        if kv_cache_position_ids is None:
            return self._get_all_cache_handles(layer_id, batch_id)
        
        selected_handles = []
        for position in kv_cache_position_ids:
            handle = self._get_cache_handle(position, layer_id, batch_id)
            if handle:
                selected_handles.append(handle)
        
        logger.info(f"Selected {len(selected_handles)} cache handles for {len(kv_cache_position_ids)} positions at layer={layer_id}, batch={batch_id}")
        return selected_handles

    @contextlib.contextmanager
    def use_cache(self, *handles: Handle):
        """
        Context manager for using cache handles
        Yields cache tensors for given handles
        """
        if not handles:
            yield []
            return
        
        offload_logger.info(f"Using cache handles: {handles}")
        offload_logger.info(f"Number of handles: {len(handles)}")
        
        with self.cache.use_cache(*handles) as cache_tensors:
            offload_logger.info(f"Cache tensors retrieved: {len(cache_tensors)} tensors")
            yield cache_tensors

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics and information"""
        return {
            'cache_size_bytes': self.cache_size,
            'current_size_bytes': self.cache.current_size_bytes,
            'bytes_left': self.cache.bytes_left,
            'stats': self.stats.copy(),
            'hierarchy_layers': len(self.cache_hierarchy),
            'total_handles': len(self.cache._unified_caches)
        }

    def _calculate_cache_size(self, unified_cache: UnifiedCache) -> int:
        """Calculate size of UnifiedCache in bytes"""
        if unified_cache.past_key_value is None:
            return 0
        
        total_size = 0
        for tensor in unified_cache.past_key_value:
            if isinstance(tensor, torch.Tensor):
                total_size += tensor.numel() * tensor.element_size()
        
        return total_size

    def _get_cache_handle(self, position: int, layer_id: int, batch_id: int) -> Optional[Handle]:
        """Get cache handle for specific position, layer, and batch"""
        offload_logger.info(f"Looking for cache handle - position:{position}, layer:{layer_id}, batch:{batch_id}")
        
        if layer_id in self.cache_hierarchy:
            if batch_id in self.cache_hierarchy[layer_id]:
                if position in self.cache_hierarchy[layer_id][batch_id]:
                    handle = self.cache_hierarchy[layer_id][batch_id][position]
                    offload_logger.info(f"Found handle: {handle}")
                    return handle
                else:
                    # Show all available positions for this layer and batch
                    available_positions = list(self.cache_hierarchy[layer_id][batch_id].keys())
                    offload_logger.info(f"Position {position} not found, available positions: {available_positions}")
            else:
                # Show all available batches for this layer
                available_batches = list(self.cache_hierarchy[layer_id].keys())
                offload_logger.info(f"Batch {batch_id} not found, available batches: {available_batches}")
        else:
            # Show all available layers
            available_layers = list(self.cache_hierarchy.keys())
            offload_logger.info(f"Layer {layer_id} not found, available layers: {available_layers}")
        
        offload_logger.warning("Cache handle not found")
        return None

    def _get_all_cache_handles(self, layer_id: int, batch_id: int) -> List[Handle]:
        """Get all cache handles for a specific layer and batch"""
        all_handles = []
        if (layer_id in self.cache_hierarchy and 
            batch_id in self.cache_hierarchy[layer_id]):
            for handle in self.cache_hierarchy[layer_id][batch_id].values():
                all_handles.append(handle)
        return all_handles

    def _update_cache_stats(self, cache_size: int, layer_id: int, device_info: DeviceInfo):
        """Update cache statistics"""
        self.stats['total_bytes_stored'] += cache_size
        offload_logger.info(f"Updated cache stats - size: {cache_size / 1024:.1f} KB, layer: {layer_id}, device: {device_info.device_id}")


class BlockCacheAdapter:
    """
    Adapter class for connecting block.py cache management with KVCacheManager
    Provides conversion between block cache format and KVCache format
    """
    
    def __init__(self, cache_manager: KVCacheManager):
        self.cache_manager = cache_manager
    
    def register_block_cache(self, layer_id: int, batch_id: int, 
                           cache_home, cache_read_buf, cache_write_buf):
        """Register block cache with KVCacheManager"""
        # This is a placeholder for future implementation
        offload_logger.info(f"Registered block cache for layer {layer_id}, batch {batch_id}")
    
    def sync_block_cache_to_manager(self, layer_id: int, batch_id: int, position: int):
        """Sync block cache to KVCacheManager"""
        # Convert block cache to KVCache format
        block_cache = None  # Get from block system
        if block_cache:
            kvs = self._convert_block_cache_to_kvs(block_cache)
            self.cache_manager.add_cache(kvs, position, layer_id, batch_id)
    
    def sync_manager_cache_to_block(self, layer_id: int, batch_id: int, position: int):
        """Sync KVCacheManager cache to block"""
        # Load from KVCacheManager
        unified_cache = self.cache_manager.load_cache(position, layer_id, batch_id)
        if unified_cache:
            # Convert to block cache format
            block_cache = self._convert_kvs_to_block_cache(unified_cache.past_key_value)
            # Store in block system
            offload_logger.info(f"Synced manager cache to block for layer {layer_id}, position {position}")
    
    def _convert_block_cache_to_kvs(self, block_cache) -> KVCache:
        """Convert block cache to KVCache format"""
        # Placeholder implementation
        # TODO: Implement actual conversion logic
        return KVCache(
            kvs=block_cache,
            device=KVCacheMetadata(
                device=None,  # Will be set by caller
                offloaded=False
            )
        )
    
    def _convert_kvs_to_block_cache(self, cache_tensors) -> tuple:
        """Convert KVCache format to block cache format"""
        # Placeholder implementation
        # TODO: Implement actual conversion logic
        return cache_tensors




if __name__ == "__main__":
    test_kv_cache_manager()
        
        
