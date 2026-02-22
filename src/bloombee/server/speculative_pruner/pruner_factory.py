
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from bloombee.server.speculative_pruner.simple_probability_pruner import SimpleProbabilityPruner
from bloombee.server.speculative_pruner.adaptive_neural_pruner import AdaptiveNeuralPruner
from bloombee.server.speculative_pruner.utils import PruningMethod, PruningConfig
from bloombee.server.speculative_pruner.pruner_interface import PrunerInterface

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
            return SimpleProbabilityPruner(hidden_size, vocab_size, config, tie_weights)
        elif method == PruningMethod.ADAPTIVE_NEURAL:
            return AdaptiveNeuralPruner(hidden_size, vocab_size, 64, 'cuda', config)
        else:
            raise ValueError(f"Unknown pruning method: {method}")


