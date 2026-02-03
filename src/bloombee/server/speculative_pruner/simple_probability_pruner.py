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
        """单个序列的 parent position 查找"""
        for j in range(i-1, -1, -1):
            if mask[i, j + prefix] == True:
                return j
        return i
    
    def _get_parent_postion_batched(self, i, mask, prefix, batch_idx):
        """batch 版本的 parent position 查找"""
        for j in range(i-1, -1, -1):
            if mask[batch_idx, i, j + prefix] == True:
                return j
        return i
    
    def prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: Union[List[int], torch.Tensor],
        tree_attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prune branches based on probability threshold
        Process nodes sequentially in depth-first order
        
        支持 batch 处理
        
        Args:
            middle_hidden_states: [B, seq_len, hidden_size] - hidden states in depth-first order
            draft_tokens: [B, seq_len] Token IDs in depth-first order  
            tree_attention_mask: [B, seq_len, total_len] - encodes tree structure
            
        Returns:
            keep_indices: [B, max_keep_len] 保留的索引，padding 用 -1
            其他元数据
        """
        
        # 处理输入维度
        if middle_hidden_states.dim() == 2:
            # 单序列情况，扩展为 batch
            middle_hidden_states = middle_hidden_states.unsqueeze(0)
            tree_attention_mask = tree_attention_mask.unsqueeze(0) if tree_attention_mask.dim() == 2 else tree_attention_mask
            if isinstance(draft_tokens, list):
                draft_tokens = [draft_tokens]
            elif isinstance(draft_tokens, torch.Tensor) and draft_tokens.dim() == 1:
                draft_tokens = draft_tokens.unsqueeze(0)
        
        batch_size = middle_hidden_states.shape[0]
        seq_len = middle_hidden_states.shape[1]
        device = middle_hidden_states.device
        
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        # Get middle layer logits and probabilities: [B, seq_len, vocab_size]
        middle_logits = self.lm_head(middle_hidden_states)
        
        # 转换 draft_tokens 为 tensor
        if isinstance(draft_tokens, list):
            if isinstance(draft_tokens[0], list):
                # List of lists
                draft_tokens = torch.tensor(draft_tokens, device=device)
            else:
                # Single list
                draft_tokens = torch.tensor(draft_tokens, device=device).unsqueeze(0)
        
        # 存储每个 batch 的结果
        batch_keep_indices = []
        batch_prune_indices = []
        batch_scores = []
        batch_keep_masks = []
        
        for b in range(batch_size):
            # Initialize keep mask (all True initially)
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            
            # Track which nodes are discarded (for skipping descendants)
            discarded = torch.zeros(seq_len, dtype=torch.bool, device=device)
            
            # Store scores for all nodes
            scores = torch.zeros(seq_len, device=device)
            
            # Process each node in depth-first order
            for i in range(seq_len):
                if i == 0:
                    keep_mask[0] = True
                    scores[i] = 1.0
                    continue
                
                # Skip if already discarded by ancestor
                if discarded[i]:
                    keep_mask[i] = False
                    scores[i] = 0.0
                    continue
                
                # Get token probability
                parent_position = self._get_parent_postion_batched(i, tree_attention_mask, prefix_len, b)
                logits_at_pos = middle_logits[b, parent_position]
                probs = F.softmax(logits_at_pos, dim=-1)
                topk = 50

                # 取 top-50 token ids
                topk_ids = torch.topk(
                    probs,
                    k=min(topk, self.vocab_size),
                    dim=-1
                ).indices

                # 判断 draft token 是否在 topk
                draft_id = draft_tokens[b, i].item()
                label = 1.0 if draft_id in topk_ids.tolist() else 0.0
                
                draft_prob = probs[draft_id].item()
                scores[i] = draft_prob
                
                # Check if score meets threshold
                if label == 0.0:
                    keep_mask[i] = False
                    discarded[i] = True
                    
                    # Mark all descendants as discarded
                    for j in range(i + 1, seq_len):
                        if tree_attention_mask[b, j, i + prefix_len] == 1:
                            discarded[j] = True
                            keep_mask[j] = False
            
            # Get final indices for this batch
            keep_indices = torch.where(keep_mask)[0].tolist()
            prune_indices = torch.where(~keep_mask)[0].tolist()
            
            batch_keep_indices.append(keep_indices)
            batch_prune_indices.append(prune_indices)
            batch_scores.append(scores)
            batch_keep_masks.append(keep_mask)
        
        # Padding keep_indices to same length with -1
        max_keep_len = max(len(indices) for indices in batch_keep_indices)
        
        padded_keep_indices = torch.full(
            (batch_size, max_keep_len), 
            -1, 
            dtype=torch.long, 
            device=device
        )
        
        for b, indices in enumerate(batch_keep_indices):
            if len(indices) > 0:
                padded_keep_indices[b, :len(indices)] = torch.tensor(indices, device=device)
        
        # Padding prune_indices
        max_prune_len = max(len(indices) for indices in batch_prune_indices) if batch_prune_indices else 0
        if max_prune_len > 0:
            padded_prune_indices = torch.full(
                (batch_size, max_prune_len), 
                -1, 
                dtype=torch.long, 
                device=device
            )
            for b, indices in enumerate(batch_prune_indices):
                if len(indices) > 0:
                    padded_prune_indices[b, :len(indices)] = torch.tensor(indices, device=device)
        else:
            padded_prune_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        # Stack scores and masks
        stacked_scores = torch.stack(batch_scores, dim=0)  # [B, seq_len]
        stacked_keep_masks = torch.stack(batch_keep_masks, dim=0)  # [B, seq_len]
        
        # 计算有效长度
        valid_lengths = (padded_keep_indices >= 0).sum(dim=1)  # [B]
        
        return {
            'keep_indices': padded_keep_indices,  # [B, max_keep_len]，padding 为 -1
            'prune_indices': padded_prune_indices,  # [B, max_prune_len]，padding 为 -1
            'keep_probs': stacked_scores,  # [B, seq_len]
            'keep_mask': stacked_keep_masks,  # [B, seq_len]
            'valid_lengths': valid_lengths,  # [B] 每个 batch 的有效 keep 数量
            'metadata': {
                'middle_logits': middle_logits,
                'avg_score': stacked_scores[stacked_keep_masks].mean().item() if stacked_keep_masks.any() else 0.0,
            }
        }
    
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