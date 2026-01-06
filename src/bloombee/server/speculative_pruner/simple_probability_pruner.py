import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from huggingface_hub import hf_hub_download

from bloombee.server.speculative_pruner.pruner_interface import PrunerInterface
from bloombee.server.speculative_pruner.utils import PruningConfig
from bloombee.server.speculative_pruner.mid_layer_LM_head import MidLMHead

class SimpleProbabilityPruner(PrunerInterface):
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        config: PruningConfig,
        device: str = 'cuda',
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device
        
        # LM head for getting probabilities
        self.lm_head = MidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to("cuda")
        lm_head_weights_path = hf_hub_download(
            repo_id="xxiong59/lm-head-for-speculative-pruning",
            filename="lm_head_weights_15.pt",
            cache_dir="./cache"
        )
        lm_head_checkpoint = torch.load(lm_head_weights_path, map_location="cuda")
        if 'model_state_dict' in lm_head_checkpoint:
            self.lm_head.load_state_dict(lm_head_checkpoint['model_state_dict'])
        else:
            self.lm_head.load_state_dict(lm_head_checkpoint)
        self.lm_head.requires_grad_(False)
        self.lm_head.to(dtype=torch.float16)
        
        # Statistics
        self.total_branches = 0
        self.pruned_branches = 0
        self.correct_prunes = 0
        
    def _get_parent_postion(self, i, mask, prefix):
        for j in range(i-1, -1, -1):
            if mask[0, i, j + prefix] == True:
                return j
        return i
    
    def prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prune branches based on probability threshold
        Process nodes sequentially in depth-first order
        
        Args:
            middle_hidden_states: [seq_len, hidden_size] - hidden states in depth-first order
            draft_tokens: Token IDs in depth-first order  
            tree_attention_mask: [seq_len, seq_len] - encodes tree structure
        """
        
        seq_len = middle_hidden_states.shape[1]
        # logger.info(f"middle_hidden_states: {middle_hidden_states.shape}")
        
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        # Get middle layer logits and probabilities
        middle_logits = self.lm_head(middle_hidden_states)
        # probs = F.softmax(middle_logits, dim=-1)
        
        # Initialize keep mask (all True initially)
        keep_mask = torch.ones(seq_len, dtype=torch.bool)
        
        # Track which nodes are discarded (for skipping descendants)
        discarded = torch.zeros(seq_len, dtype=torch.bool)
        
        # Store scores for all nodes (for statistics and fallback)
        scores = torch.zeros(seq_len)
        
        # Process each node in depth-first order
        for i in range(seq_len):
            if i == 0:
                keep_mask[0] = True
                scores[i] = 1.0
                continue
            
            # Skip if already discarded by ancestor
            if discarded[i]:
                keep_mask[i] = False
                scores[i] = 0.0  # Set score to 0 for discarded nodes
                continue
            
            # logger.info(f"draft_tokens[i]: {draft_tokens[i]}")
            
            # Get token probability
            parent_postion = self._get_parent_postion(i, tree_attention_mask, prefix_len)
            # logger.info(f"xiongxu i : {i}, parent_postion: {parent_postion}")
            logits_at_pos = middle_logits[0, parent_postion]
            # logger.info(f"xiongxu i : {i}, logits_at_pos: {logits_at_pos}")
            probs = F.softmax(logits_at_pos, dim=-1)
            # logger.info(f"xiongxu i : {i}, probs: {probs}")
            topk = 50

            # 取 top-50 token ids（在 parent 的分布上）
            topk_ids = torch.topk(
                probs,
                k=min(topk, self.vocab_size),
                dim=-1
            ).indices  # shape: [topk]

            # 判断 draft token 是否在 topk
            label = 1.0 if draft_tokens[i] in topk_ids.tolist() else 0.0
            
            draft_id = draft_tokens[i]        # int
            draft_prob = probs[draft_id].item()
            # logger.info(f"xiongxu [node {i}] draft_token={draft_id}, prob={draft_prob:.6f}")
            
            # Check if score meets threshold
            if label == 0.0:
                keep_mask[i] = False
                discarded[i] = True
                
                # Mark all descendants as discarded
                # Descendants are nodes j > i where j can attend to i
                for j in range(i + 1, seq_len):
                    if tree_attention_mask[0, j, i + prefix_len] == 1:
                        discarded[j] = True
                        keep_mask[j] = False
        
        # Get final indices
        keep_indices = torch.where(keep_mask)[0].tolist()
        prune_indices = torch.where(~keep_mask)[0].tolist()
        
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'keep_probs': scores.tolist(),
            'keep_mask': keep_mask,
            'metadata': {
                'middle_logits': middle_logits,
                'avg_score': scores[keep_mask].mean().item() if keep_mask.any() else 0.0,
            }
        }
    
    def _create_pruned_attention_mask(
        self,
        original_mask: torch.Tensor,
        keep_indices: List[int]
    ) -> torch.Tensor:
        """Create new attention mask for kept nodes only"""
        new_len = len(keep_indices)
        new_mask = torch.zeros(new_len, new_len, dtype=original_mask.dtype)
        
        for new_i, old_i in enumerate(keep_indices):
            for new_j, old_j in enumerate(keep_indices):
                new_mask[new_i, new_j] = original_mask[old_i, old_j]
        
        return new_mask
    
    def get_metrics(self) -> Dict[str, float]:
        """Get pruning metrics"""
        prune_rate = self.pruned_branches / max(self.total_branches, 1)
        accuracy = self.correct_prunes / max(self.pruned_branches, 1)
        
        return {
            'prune_rate': prune_rate,
            'accuracy': accuracy,
            'total_branches': self.total_branches,
            'pruned_branches': self.pruned_branches
        }