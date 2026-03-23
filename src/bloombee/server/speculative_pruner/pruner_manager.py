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
        self.pruner = None
        self._pruner_unavailable = False
        self._pruner_unavailable_reason = None
        
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

    def _ensure_pruner(self):
        if self.pruner is not None:
            return self.pruner
        if self._pruner_unavailable:
            return None

        try:
            self.pruner = SpeculativePrunerFactory.create_pruner(
                self.config.method,
                self.hidden_size,
                self.vocab_size,
                self.config,
            )
        except Exception as e:
            self._pruner_unavailable = True
            self._pruner_unavailable_reason = repr(e)
            logger.warning(
                "Disabling speculative pruner because initialization failed: %s",
                e,
                exc_info=True,
            )
            return None

        return self.pruner

    def _build_keep_all_result(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: Union[List[int], torch.Tensor],
    ) -> Dict[str, Any]:
        if middle_hidden_states.dim() == 2:
            middle_hidden_states = middle_hidden_states.unsqueeze(0)

        batch_size, seq_len = middle_hidden_states.shape[:2]
        device = middle_hidden_states.device
        keep_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        prune_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        keep_probs = torch.ones((batch_size, seq_len), dtype=middle_hidden_states.dtype, device=device)
        keep_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        valid_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'keep_probs': keep_probs,
            'keep_mask': keep_mask,
            'valid_lengths': valid_lengths,
            'metadata': {
                'fallback': 'keep_all',
                'reason': self._pruner_unavailable_reason or 'pruner_unavailable',
            },
        }

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
        current_pruner = self._ensure_pruner() if keep_stats else self.pruner
        old_metrics = current_pruner.get_metrics() if keep_stats and current_pruner is not None else None

        if isinstance(method, str):
            method = PruningMethod(method)
        self.config.method = method
        self.pruner = None
        self._pruner_unavailable = False
        self._pruner_unavailable_reason = None
        pruner = self._ensure_pruner()
        
        if keep_stats and old_metrics and pruner is not None:
            # Transfer relevant statistics
            if hasattr(pruner, 'acceptance_history'):
                pruner.acceptance_history = deque(old_metrics.get('acceptance_history', []), maxlen=100)
    
    def prune_speculation_tree(
        self,
        norm_hidden_states: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        pruner = self._ensure_pruner()
        if pruner is None:
            return self._build_keep_all_result(norm_hidden_states, draft_tokens)

        kwargs = {}
        results = pruner.prune_branches(
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
        pruner = self._ensure_pruner()
        if pruner is not None and hasattr(pruner, 'train_step'):
            with torch.enable_grad():
                pruner.train_step(middle_norm_hidden_states, final_logits, attention_mask, draft_tokens)
                self.iteration = self.iteration + 1
                
    def train_lm_head(self, middle_hidden_states, final_hidden_states):
        trainer = self._ensure_lm_head_trainer()
        if trainer is not None:
            with torch.enable_grad():
                trainer.train_step(middle_hidden_states, final_hidden_states)
                self.iteration = self.iteration + 1
