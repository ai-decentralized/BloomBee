"""
Cache coordinator for managing cache operations across model layers
Provides unified interface for cache management and device allocation
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from bloombee.data_structures import KVCache, KVCacheMetadata, Handle, UnifiedCache, DeviceInfo
from bloombee.server.memory_cache_manager import KVCacheManager

# Create dedicated offloading debug logger
offload_logger = logging.getLogger('bloombee.offloading')
offload_logger.setLevel(logging.INFO)


def get_device_allocation_from_policy(policy, fallback_device: str = 'cuda:0') -> Tuple[str, str]:
    """
    统一的设备分配工具函数
    
    Args:
        policy: Policy对象，包含缓存配置
        fallback_device: 当没有policy时的默认设备
        
    Returns:
        (device_type, device_id): 设备类型和设备ID
    """
    if policy is not None:
        if policy.cache_gpu_percent == 100:
            device_type = 'gpu'
            device_id = 'cuda:0'
        elif policy.cache_cpu_percent == 100:
            device_type = 'cpu'
            device_id = 'cpu'
        elif policy.cache_disk_percent == 100:
            device_type = 'disk'
            device_id = '/tmp/disk_cache'
        else:
            # Mixed allocation - use CPU as primary for offloading
            device_type = 'cpu'
            device_id = 'cpu'
    else:
        # 如果没有policy，使用默认设备
        if fallback_device.startswith('cuda'):
            device_type = 'gpu'
            device_id = fallback_device
        else:
            device_type = 'cpu'
            device_id = 'cpu'
    
    return device_type, device_id


def create_device_info_from_policy(policy, fallback_device: str = 'cuda:0', 
                                 compression_config=None) -> DeviceInfo:
    """
    根据policy创建DeviceInfo对象
    
    Args:
        policy: Policy对象，包含缓存配置
        fallback_device: 当没有policy时的默认设备
        compression_config: 压缩配置
        
    Returns:
        DeviceInfo对象
    """
    device_type, device_id = get_device_allocation_from_policy(policy, fallback_device)
    
    return DeviceInfo(
        device_type=device_type,
        device_id=device_id,
        compression_config=compression_config,
        offloaded=(device_type != 'gpu')
    )


class CacheCoordinator:
    """
    Lightweight adapter for KVCacheManager
    Provides layer registration and simplified interface without duplicating functionality
    """
    
    def __init__(self, cache_manager: KVCacheManager):
        self.cache_manager = cache_manager
        self._layer_registry: Dict[int, Dict[str, Any]] = {}
        
        offload_logger.info("Initializing CacheCoordinator")
        offload_logger.info(f"Cache manager: {type(cache_manager).__name__}")
    
    def register_layer(self, layer_id: int, layer_info: Dict[str, Any] = None):
        """Register a model layer with the coordinator"""
        if layer_info is None:
            layer_info = {}
        
        self._layer_registry[layer_id] = {
            'info': layer_info,
            'registered': True
        }
        
        offload_logger.info(f"Registered layer {layer_id} with coordinator")
    
    def unregister_layer(self, layer_id: int):
        """Unregister a model layer from the coordinator"""
        if layer_id in self._layer_registry:
            del self._layer_registry[layer_id]
            offload_logger.info(f"Unregistered layer {layer_id} from coordinator")
    
    def load_cache_for_layer(self, layer_id: int, position: int, 
                           target_device: str = 'cuda:0', batch_id: int = 0) -> Optional[UnifiedCache]:
        """
        Load cache for a specific layer and position
        Delegates to KVCacheManager.load_cache()
        """
        offload_logger.info(f"CacheCoordinator.load_cache_for_layer - layer:{layer_id}, position:{position}")
        
        # 直接委托给KVCacheManager
        return self.cache_manager.load_cache(position, layer_id, batch_id, target_device)
    
    def store_cache_for_layer(self, layer_id: int, position: int, 
                            past_key_value: Tuple[torch.Tensor, ...],
                            device: torch.device, batch_id: int = 0) -> Optional[Handle]:
        """
        Store cache for a specific layer and position
        Delegates to KVCacheManager.store_cache()
        """
        offload_logger.info(f"CacheCoordinator.store_cache_for_layer - layer:{layer_id}, position:{position}")
        
        # 验证位置一致性 - 更智能的位置处理
        expected_position = self._get_expected_position(layer_id)
        
        # 如果是prefill阶段（position=0），不需要修正
        if position == 0:
            offload_logger.info(f"Prefill阶段 - 位置: {position}, 层: {layer_id}")
        elif position != expected_position:
            offload_logger.warning(f"Position mismatch: expected {expected_position}, got {position}")
            # 只有在非prefill阶段才修正位置
            if expected_position > 0:
                position = expected_position
                offload_logger.info(f"位置已修正为: {position}")
        
        # 创建UnifiedCache
        # 根据policy决定设备分配，但优先考虑张量的实际位置
        if hasattr(self.cache_manager, 'policy') and self.cache_manager.policy is not None:
            # 检查张量的实际位置
            if past_key_value and len(past_key_value) > 0:
                first_tensor = past_key_value[0]
                if isinstance(first_tensor, torch.Tensor):
                    actual_device = str(first_tensor.device)
                    offload_logger.info(f"张量实际位置: {actual_device}")
                    
                    # 根据policy决定目标设备
                    target_device_type, target_device_id = get_device_allocation_from_policy(
                        self.cache_manager.policy, str(device)
                    )
                    offload_logger.info(f"Policy要求的目标设备: {target_device_id}")
                    
                    # 如果张量位置与policy要求不符，需要同步
                    if actual_device != target_device_id:
                        offload_logger.info(f"需要同步张量从 {actual_device} 到 {target_device_id}")
                        
                        # 同步张量到目标设备
                        synced_tensors = []
                        for i, tensor in enumerate(past_key_value):
                            if isinstance(tensor, torch.Tensor):
                                if str(tensor.device) != target_device_id:
                                    synced_tensor = tensor.to(target_device_id, non_blocking=True)
                                    offload_logger.info(f"同步张量{i}: {tensor.device} -> {synced_tensor.device}")
                                else:
                                    synced_tensor = tensor
                                    offload_logger.info(f"张量{i}已在目标设备上，跳过同步")
                                synced_tensors.append(synced_tensor)
                            else:
                                synced_tensors.append(tensor)
                        
                        # 使用同步后的张量
                        past_key_value = tuple(synced_tensors)
                        actual_device = target_device_id
                        offload_logger.info(f"同步完成，张量现在在: {actual_device}")
                    
                    # 创建设备信息
                    device_info = DeviceInfo(
                        device_type=target_device_type,
                        device_id=target_device_id,
                        compression_config=self.cache_manager.policy.comp_cache_config if self.cache_manager.policy.compress_cache else None,
                        offloaded=(target_device_type != 'gpu')
                    )
                else:
                    device_info = create_device_info_from_policy(self.cache_manager.policy, str(device))
            else:
                device_info = create_device_info_from_policy(self.cache_manager.policy, str(device))
        else:
            # 如果没有policy，使用张量的实际位置
            if past_key_value and len(past_key_value) > 0:
                first_tensor = past_key_value[0]
                if isinstance(first_tensor, torch.Tensor):
                    actual_device = str(first_tensor.device)
                    device_info = DeviceInfo(
                        device_type=actual_device.split(':')[0] if ':' in actual_device else actual_device,
                        device_id=actual_device,
                        compression_config=None,
                        offloaded=(actual_device != 'cuda:0')
                    )
                else:
                    device_info = DeviceInfo(
                        device_type=device.type,
                        device_id=str(device),
                        compression_config=None,
                        offloaded=(device.type != 'gpu')
                    )
            else:
                device_info = DeviceInfo(
                    device_type=device.type,
                    device_id=str(device),
                    compression_config=None,
                    offloaded=(device.type != 'gpu')
                )
        
        unified_cache = UnifiedCache(
            past_key_value=past_key_value,
            device_info=device_info
        )
        
        # 直接委托给KVCacheManager
        handle = self.cache_manager.store_cache(unified_cache, position, layer_id, batch_id)
        
        # 更新层状态
        if handle is not None:
            self._update_layer_position(layer_id, position, handle)
            offload_logger.info(f"成功存储缓存 - 位置:{position}, 层:{layer_id}, 句柄:{handle}, 设备:{device_info.device_type} ({device_info.device_id})")
        
        return handle
    
    def update_cache_for_layer(self, layer_id: int, position: int,
                             new_past_key_value: Tuple[torch.Tensor, ...],
                             device: torch.device, batch_id: int = 0) -> Optional[Handle]:
        """
        Update existing cache for a specific layer and position
        Delegates to KVCacheManager.update_cache()
        """
        offload_logger.info(f"CacheCoordinator.update_cache_for_layer - layer:{layer_id}, position:{position}")
        
        # 直接委托给KVCacheManager
        return self.cache_manager.update_cache(new_past_key_value, position, layer_id, batch_id)
    
    def get_layer_info(self, layer_id: int) -> Dict[str, Any]:
        """Get layer registration information"""
        if layer_id not in self._layer_registry:
            return {}
        
        return {
            'registered': True,
            'layer_info': self._layer_registry[layer_id]['info']
        }
    
    def get_registered_layers(self) -> List[int]:
        """Get list of registered layer IDs"""
        return list(self._layer_registry.keys())
    
    def is_layer_registered(self, layer_id: int) -> bool:
        """Check if a layer is registered"""
        return layer_id in self._layer_registry
    
    def get_cache_manager(self) -> KVCacheManager:
        """Get the underlying cache manager"""
        return self.cache_manager
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from underlying cache manager"""
        return self.cache_manager.get_cache_info()
    
    def _get_expected_position(self, layer_id: int) -> int:
        """Get expected position for a layer based on current state"""
        if layer_id in self._layer_registry:
            return self._layer_registry[layer_id].get('last_position', 0) + 1
        return 0
    
    def _update_layer_position(self, layer_id: int, position: int, handle: Handle):
        """Update layer position tracking"""
        if layer_id in self._layer_registry:
            self._layer_registry[layer_id]['last_position'] = position
            self._layer_registry[layer_id]['last_handle'] = handle
            if 'cache_count' not in self._layer_registry[layer_id]:
                self._layer_registry[layer_id]['cache_count'] = 0
            self._layer_registry[layer_id]['cache_count'] += 1
            
            offload_logger.info(f"Updated layer {layer_id} position to {position} with handle {handle}")
    
    def get_layer_cache_info(self, layer_id: int) -> Dict[str, Any]:
        """Get detailed cache information for a specific layer"""
        if layer_id not in self._layer_registry:
            return {}
        
        info = self._layer_registry[layer_id].copy()
        info['expected_position'] = self._get_expected_position(layer_id)
        
        # Get available positions from cache manager
        if hasattr(self.cache_manager, '_position_tracker') and layer_id in self.cache_manager._position_tracker:
            info['available_positions'] = list(self.cache_manager._position_tracker[layer_id].keys())
        
        return info


class ModelCacheInterface:
    """
    Interface for model layers to interact with cache without direct dependency
    Provides a clean abstraction layer
    """
    
    def __init__(self, coordinator: CacheCoordinator):
        self.coordinator = coordinator
    
    def load_cache(self, layer_id: int, position: int, 
                  target_device: str = 'cuda:0', batch_id: int = 0) -> Optional[UnifiedCache]:
        """Load cache for model layer"""
        return self.coordinator.load_cache_for_layer(layer_id, position, target_device, batch_id)
    
    def store_cache(self, layer_id: int, position: int,
                   past_key_value: Tuple[torch.Tensor, ...],
                   device: torch.device, batch_id: int = 0) -> Optional[Handle]:
        """Store cache for model layer"""
        return self.coordinator.store_cache_for_layer(layer_id, position, past_key_value, device, batch_id)
    
    def update_cache(self, layer_id: int, position: int,
                    new_past_key_value: Tuple[torch.Tensor, ...],
                    device: torch.device, batch_id: int = 0) -> Optional[Handle]:
        """Update cache for model layer"""
        return self.coordinator.update_cache_for_layer(layer_id, position, new_past_key_value, device, batch_id)
    
    def register_layer(self, layer_id: int, layer_info: Dict[str, Any] = None):
        """Register model layer with coordinator"""
        self.coordinator.register_layer(layer_id, layer_info)
    
    def unregister_layer(self, layer_id: int):
        """Unregister model layer from coordinator"""
        self.coordinator.unregister_layer(layer_id)
    
    def get_layer_info(self, layer_id: int) -> Dict[str, Any]:
        """Get layer registration information"""
        return self.coordinator.get_layer_info(layer_id)
    
    def is_layer_registered(self, layer_id: int) -> bool:
        """Check if a layer is registered"""
        return self.coordinator.is_layer_registered(layer_id)
    
    def get_layer_cache_info(self, layer_id: int) -> Dict[str, Any]:
        """Get detailed cache information for a layer"""
        return self.coordinator.get_layer_cache_info(layer_id)


# Global cache coordinator instance
_global_cache_coordinator: Optional[CacheCoordinator] = None
_global_cache_interface: Optional[ModelCacheInterface] = None


def get_cache_coordinator() -> Optional[CacheCoordinator]:
    """Get the global cache coordinator instance"""
    return _global_cache_coordinator


def get_cache_interface() -> Optional[ModelCacheInterface]:
    """Get the global cache interface for model layers"""
    return _global_cache_interface


def set_cache_coordinator(cache_manager: KVCacheManager):
    """Set the global cache coordinator"""
    global _global_cache_coordinator, _global_cache_interface
    
    _global_cache_coordinator = CacheCoordinator(cache_manager)
    _global_cache_interface = ModelCacheInterface(_global_cache_coordinator)
    
    offload_logger.info("Global cache coordinator initialized")
    offload_logger.info(f"Cache manager: {type(cache_manager).__name__}")


def clear_cache_coordinator():
    """Clear the global cache coordinator"""
    global _global_cache_coordinator, _global_cache_interface
    
    _global_cache_coordinator = None
    _global_cache_interface = None
    
    offload_logger.info("Global cache coordinator cleared") 