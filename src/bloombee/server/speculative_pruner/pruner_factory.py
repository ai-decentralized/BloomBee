
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

class PruningMethod(Enum):
    """Enumeration of available pruning methods"""
    SIMPLE_PROBABILITY = "simple_probability"
    ADAPTIVE_NEURAL = "adaptive_neural"


@dataclass
class PruningConfig:
    """Configuration for pruning strategies"""
    method: PruningMethod = PruningMethod.SIMPLE_PROBABILITY
    
    # Simple probability method config
    simple_threshold: float = 0.3
    simple_top_k: int = 5
    simple_use_entropy: bool = False
    
    # Adaptive neural method config
    neural_threshold: float = 0.5
    neural_hidden_size: int = 128
    neural_dropout: float = 0.1
    neural_learning_rate: float = 1e-4
    neural_speedup_weight: float = 0.1
    
    # Shared config
    max_branches: int = 32
    min_keep_branches: int = 1
    min_keep_nodes: int = 5
    enable_caching: bool = True
    cache_size: int = 1000


class PrunerInterface(ABC):
    """Abstract base class for all pruning strategies"""
    
    @abstractmethod
    def prune_branches(
        self,
        logits: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prune branches based on the specific strategy
        
        Returns:
            Dictionary containing:
            - keep_indices: List of branch indices to keep
            - prune_indices: List of branch indices to prune
            - keep_probs: Probabilities for each branch
            - metadata: Additional strategy-specific information
        """
        pass
    
    @abstractmethod
    def update_statistics(self, feedback: Dict[str, Any]):
        """Update internal statistics based on pruning outcomes"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        pass


class SpeculativePrunerFactory:
    """Factory class for creating pruners with Bloombee integration"""
    
    @staticmethod
    def create_pruner(
        method: Union[str, PruningMethod],
        hidden_size: int,
        vocab_size: int,
        config: Optional[PruningConfig] = None,
        tie_weights: Optional[nn.Linear] = None
    ) -> PrunerInterface:
        """
        Create a pruner instance based on the specified method
        
        Args:
            method: Pruning method to use
            hidden_size: Hidden size of the model
            vocab_size: Vocabulary size
            config: Optional configuration object
            tie_weights: Optional weights to tie with main LM head
            
        Returns:
            Pruner instance implementing PrunerInterface
        """
        
        if config is None:
            config = PruningConfig()
        
        if isinstance(method, str):
            method = PruningMethod(method)
        
        config.method = method
        
        if method == PruningMethod.SIMPLE_PROBABILITY:
            logger.info("Creating Simple Probability Pruner")
            return SimpleProbabilityPruner(hidden_size, vocab_size, config, tie_weights)
        elif method == PruningMethod.ADAPTIVE_NEURAL:
            logger.info("Creating Adaptive Neural Pruner")
            return AdaptiveNeuralPruner(hidden_size, vocab_size, 64, 'cuda', config)
        else:
            raise ValueError(f"Unknown pruning method: {method}")


