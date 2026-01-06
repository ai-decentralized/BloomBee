import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import os

class SpeculativePrunerManager:
    """
    Manager class for handling pruning in Bloombee framework
    Provides high-level interface for speculative decoding with pruning
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        config: Optional[PruningConfig] = None,
        device: str = 'cuda'
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = config or PruningConfig()
        self.device = device
        
        # Create initial pruner
        self.pruner = BloombeePrunerFactory.create_pruner(
            self.config.method,
            hidden_size,
            vocab_size,
            self.config
        )
        
        # Metrics tracking
        self.total_tokens = 0
        self.pruned_tokens = 0
        self.speedup_factor = 1.0
        self.iteration = 0
        self.middle_states = None
        
    def switch_method(self, method: Union[str, PruningMethod], keep_stats: bool = False):
        """Switch to a different pruning method"""
        old_metrics = self.pruner.get_metrics() if keep_stats else None
        
        self.pruner = BloombeePrunerFactory.create_pruner(
            method,
            self.hidden_size,
            self.vocab_size,
            self.config
        )
        
        if keep_stats and old_metrics:
            # Transfer relevant statistics
            if hasattr(self.pruner, 'acceptance_history'):
                self.pruner.acceptance_history = deque(old_metrics.get('acceptance_history', []), maxlen=100)
        
        logger.info(f"Switched to {method} pruning method")
    
    def prune_speculation_tree(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        kwargs = {}
        if isinstance(self.pruner, AdaptiveNeuralPruner):
            kwargs['network_condition'] = self.pruner.get_network_condition()
        
        # Perform pruning
        results = self.pruner.prune_branches(
            self.middle_states,
            draft_tokens,
            tree_attention_mask,
            **kwargs
        )
        
        # Update statistics
        self.total_tokens += len(draft_tokens)
        self.pruned_tokens += len(results['prune_indices'])
        
        # Calculate speedup
        if self.total_tokens > 0:
            keep_rate = 1.0 - (self.pruned_tokens / self.total_tokens)
            # Approximate speedup based on pruning rate
            self.speedup_factor = 1.0 / max(keep_rate, 0.1)
        
        # Add manager-level metrics
        results['manager_metrics'] = {
            'total_tokens': self.total_tokens,
            'pruned_tokens': self.pruned_tokens,
            'speedup_factor': self.speedup_factor,
            'keep_rate': 1.0 - (self.pruned_tokens / self.total_tokens)
        }
        return results
    
    def save_state(self, path: str):
        state = {
            'config': {
                'method': self.config.method.value,
                'hidden_size': self.hidden_size,
                'vocab_size': self.vocab_size,
            },
            'manager_metrics': {
                'total_tokens': self.total_tokens,
                'pruned_tokens': self.pruned_tokens,
                'speedup_factor': self.speedup_factor,
                'iteration': self.iteration,
            }
        }
        
        # Save pruner-specific state if available
        if hasattr(self.pruner, 'save_model'):
            pruner_path = path.replace('.pt', '_pruner.pt')
            self.pruner.save_model(pruner_path)
            state['pruner_path'] = pruner_path
        
        torch.save(state, path)
        logger.info(f"Saved pruner manager state to {path}")
    
    def load_state(self, path: str):
        if not os.path.exists(path):
            print(f"[Pruner] No checkpoint found at {path}, starting fresh.")
            return False
        
        state = torch.load(path, map_location=self.device)
    
        # Restore manager metrics
        self.total_tokens = state['manager_metrics']['total_tokens']
        self.pruned_tokens = state['manager_metrics']['pruned_tokens']
        self.speedup_factor = state['manager_metrics']['speedup_factor']
        self.iteration = state['manager_metrics'].get('iteration', 0)
        
        # Load pruner-specific state if available
        if 'pruner_path' in state and hasattr(self.pruner, 'load_model'):
            self.pruner.load_model(state['pruner_path'])
            logger.info(f"Loaded pruner model from {state['pruner_path']}")
        
        logger.info(f"Loaded pruner manager state from {path}")
        
    def train_model(self, final_hidden_states, attention_mask, draft_tokens):
        if hasattr(self.pruner, 'train_step'):
            with torch.enable_grad():
                self.pruner.train_step(self.middle_states, final_hidden_states, attention_mask, draft_tokens)
                self.iteration = self.iteration + 1
        