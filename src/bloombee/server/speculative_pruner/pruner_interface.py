
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

class PrunerInterface(ABC):
    """Abstract base class for all pruning strategies"""
    
    @abstractmethod
    def prune_branches(
        self,
        norm_hidden_states: torch.Tensor,
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
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        pass