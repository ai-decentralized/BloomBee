import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import os
import logging

from bloombee.server.speculative_pruner.pruner_factory import SpeculativePrunerFactory
from bloombee.server.speculative_pruner.utils import PruningMethod, PruningConfig
from bloombee.server.speculative_pruner.lm_head_trainer import LM_head_trainer

logger = logging.getLogger(__name__)

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
        self.pruner = SpeculativePrunerFactory.create_pruner(
            self.config.method,
            hidden_size,
            vocab_size,
            self.config
        )
        
        # Metrics tracking
        self.total_tokens = 0
        self.pruned_tokens = 0
        self.iteration = 0
        self.middle_states = None

        # The auxiliary LM-head trainer depends on an out-of-band weight file.
        # Server startup should not fail if that file is absent because inference
        # and pruning do not require online LM-head training.
        self.lm_head_trainer = None
        self._lm_head_trainer_unavailable = False

    def _ensure_lm_head_trainer(self) -> Optional[LM_head_trainer]:
        if self.lm_head_trainer is not None:
            return self.lm_head_trainer
        if self._lm_head_trainer_unavailable:
            return None

        try:
            self.lm_head_trainer = LM_head_trainer(
                self.hidden_size,
                self.vocab_size,
                self.device,
                self.config,
            )
        except FileNotFoundError as e:
            self._lm_head_trainer_unavailable = True
            logger.warning(
                "Disabling auxiliary LM-head trainer because its weight file is missing: %s",
                e,
            )
            return None

        return self.lm_head_trainer
        
    def switch_method(self, method: Union[str, PruningMethod], keep_stats: bool = False):
        """Switch to a different pruning method"""
        old_metrics = self.pruner.get_metrics() if keep_stats else None
        
        self.pruner = SpeculativePrunerFactory.create_pruner(
            method,
            self.hidden_size,
            self.vocab_size,
            self.config
        )
        
        if keep_stats and old_metrics:
            # Transfer relevant statistics
            if hasattr(self.pruner, 'acceptance_history'):
                self.pruner.acceptance_history = deque(old_metrics.get('acceptance_history', []), maxlen=100)
    
    def prune_speculation_tree(
        self,
        norm_hidden_states: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        
        kwargs = {}
        results = self.pruner.prune_branches(
            norm_hidden_states,
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
        
        # Add manager-level metrics
        results['manager_metrics'] = {
            'total_tokens': self.total_tokens,
            'pruned_tokens': self.pruned_tokens,
            'keep_rate': keep_rate,
        }
        self.iteration = self.iteration + 1
        return results
        
    def train_model(self, middle_norm_hidden_states, final_logits, attention_mask, draft_tokens):
        if hasattr(self.pruner, 'train_step'):
            with torch.enable_grad():
                self.pruner.train_step(middle_norm_hidden_states, final_logits, attention_mask, draft_tokens)
                self.iteration = self.iteration + 1
                
    def train_lm_head(self, middle_hidden_states, final_hidden_states):
        trainer = self._ensure_lm_head_trainer()
        if trainer is not None:
            with torch.enable_grad():
                trainer.train_step(middle_hidden_states, final_hidden_states)
                self.iteration = self.iteration + 1
