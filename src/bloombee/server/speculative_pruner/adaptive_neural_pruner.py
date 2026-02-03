import torch
from huggingface_hub import hf_hub_download
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import math
import logging

from bloombee.server.speculative_pruner.mid_layer_LM_head import MidLMHead
from bloombee.server.speculative_pruner.utils import NetworkCondition

logger = logging.getLogger(__name__)

class NodePruner(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        self.quality_path = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.threshold_path = nn.Sequential(
            nn.Linear(4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, prob_features):
        quality_score = self.quality_path(prob_features).squeeze(-1)
        decision_score = quality_score
        decision_prob = torch.sigmoid(decision_score)
        
        return decision_prob, quality_score


class AdaptiveNeuralPruner:
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        neural_hidden: int = 64,
        device: str = 'cuda',
        config = None,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.config = config
        
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
        
        self.decision_net = NodePruner(hidden_size=neural_hidden).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.decision_net.parameters(), 
            lr=1e-4
        )
        
        # Training mode flag
        self.training = False
        self.ite = 0
        self.g_ite = 0
        
        self.temp_ite_count = 0
        self.atc = 0
        self.after_pruing_atc = 0
        self.keep_count = 0
        
        decision_net_weights_path = hf_hub_download(
            repo_id="xxiong59/speculative-pruning-mlp",
            filename="speculative_pruning_mlp.pt"
        )
        checkpoint = torch.load(decision_net_weights_path, map_location=device)
        if 'decision_net' in checkpoint:
            self.decision_net.load_state_dict(checkpoint['decision_net'])
        else:
            self.decision_net.load_state_dict(checkpoint)
            
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'ite' in checkpoint:
            self.ite = checkpoint['ite']
        if 'g_ite' in checkpoint:
            self.g_ite = checkpoint['g_ite']

    def _get_parent_position(self, i, mask, prefix, batch_idx=0):
        """获取 parent position，支持 batch"""
        for j in range(i - 1, -1, -1):
            if mask[batch_idx, i, j + prefix] == True:
                return j
        return i
    
    def prune_branches(
        self,
        norm_hidden_states: torch.Tensor,
        draft_tokens: Union[List[int], torch.Tensor] = None,
        tree_attention_mask: torch.Tensor = None,
        network_condition = None,
    ) -> Dict:
        """
        支持 batch 的 prune_branches
        
        Args:
            norm_hidden_states: [B, seq_len, hidden_size]
            draft_tokens: [B, seq_len] 或 List[List[int]]
            tree_attention_mask: [B, seq_len, total_len]
            network_condition: 网络条件
        
        Returns:
            keep_indices: [B, max_keep_len]，padding 为 -1
            等其他信息
        """
        # 处理输入维度
        if norm_hidden_states.dim() == 2:
            norm_hidden_states = norm_hidden_states.unsqueeze(0)
            tree_attention_mask = tree_attention_mask.unsqueeze(0) if tree_attention_mask.dim() == 2 else tree_attention_mask
            if isinstance(draft_tokens, list):
                draft_tokens = [draft_tokens]
            elif isinstance(draft_tokens, torch.Tensor) and draft_tokens.dim() == 1:
                draft_tokens = draft_tokens.unsqueeze(0)
        
        batch_size = norm_hidden_states.shape[0]
        seq_len = norm_hidden_states.shape[1]
        device = norm_hidden_states.device
        
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        if network_condition is None:
            network_condition = self.get_network_condition() or NetworkCondition.mock()
        
        # 获取 logits: [B, seq_len, vocab_size]
        logits = self.lm_head(norm_hidden_states)
        
        # 转换 draft_tokens 为 tensor
        if isinstance(draft_tokens, list):
            if isinstance(draft_tokens[0], list):
                draft_tokens = torch.tensor(draft_tokens, device=device)
            else:
                draft_tokens = torch.tensor(draft_tokens, device=device).unsqueeze(0)
        
        # 存储每个 batch 的结果
        batch_keep_indices = []
        batch_prune_indices = []
        batch_decision_probs = []
        batch_quality_scores = []
        
        for b in range(batch_size):
            # 收集该 batch 的训练数据
            prob_features_list, _ = self.collect_training_data_single(
                logits[b:b+1],
                tree_attention_mask[b:b+1],
                draft_tokens[b]
            )
            
            # 初始化 masks
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            discarded = torch.zeros(seq_len, dtype=torch.bool, device=device)
            decision_probs = torch.zeros(seq_len, device=device)
            quality_scores = torch.zeros(seq_len, device=device)
            
            for i in range(seq_len):
                if i == 0:
                    keep_mask[0] = True
                    decision_probs[0] = 1.0
                    continue
                
                if discarded[i]:
                    keep_mask[i] = False
                    decision_probs[i] = 0.0
                    continue
                
                prob_features = prob_features_list[i - 1]
                with torch.no_grad():
                    prob, quality = self.decision_net(prob_features.unsqueeze(0))
                    
                    decision_probs[i] = prob.item()
                    quality_scores[i] = quality.item()
                    keep = prob.item() > self.config.neural_threshold
                
                if not keep:
                    keep_mask[i] = False
                    discarded[i] = True
                    
                    # Discard descendants
                    for j in range(i + 1, seq_len):
                        if tree_attention_mask[b, j, i + prefix_len] == 1:
                            discarded[j] = True
                            keep_mask[j] = False
            
            keep_indices = torch.where(keep_mask)[0].tolist()
            prune_indices = torch.where(~keep_mask)[0].tolist()
            
            batch_keep_indices.append(keep_indices)
            batch_prune_indices.append(prune_indices)
            batch_decision_probs.append(decision_probs)
            batch_quality_scores.append(quality_scores)
        
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
        
        # Stack decision_probs and quality_scores
        stacked_decision_probs = torch.stack(batch_decision_probs, dim=0)  # [B, seq_len]
        stacked_quality_scores = torch.stack(batch_quality_scores, dim=0)  # [B, seq_len]
        
        # 计算有效长度
        valid_lengths = (padded_keep_indices >= 0).sum(dim=1)  # [B]
        
        return {
            'keep_indices': padded_keep_indices,  # [B, max_keep_len]
            'prune_indices': padded_prune_indices,  # [B, max_prune_len]
            'decision_probs': stacked_decision_probs,  # [B, seq_len]
            'quality_scores': stacked_quality_scores,  # [B, seq_len]
            'threshold_adjusts': 0,
            'network_condition': network_condition,
            'valid_lengths': valid_lengths,  # [B]
        }
    
    def collect_training_data_single(
        self,
        logits: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为单个 batch 收集训练数据
        
        Args:
            logits: [1, seq_len, vocab_size]
            tree_attention_mask: [1, seq_len, total_len]
            draft_tokens: [seq_len]
        """
        seq_len = draft_tokens.shape[0]
        prefix_len = tree_attention_mask.shape[2] - seq_len

        with torch.no_grad():
            prob_features_list = []
            labels_list = []

            for i in range(1, seq_len):
                parent_position = self._get_parent_position(i, tree_attention_mask, prefix_len, batch_idx=0)
                
                logits_at_pos = logits[0, parent_position]
                probs = F.softmax(logits_at_pos, dim=-1)
    
                max_prob = torch.max(probs).item()
                
                entropy = -torch.sum(torch.where(
                    probs > 1e-10,
                    probs * torch.log(probs),
                    torch.zeros_like(probs)
                )).item()
                
                log_vocab_size = math.log(float(self.vocab_size))
                if log_vocab_size > 0:
                    normalized_entropy = entropy / log_vocab_size
                else:
                    normalized_entropy = 0.0
                
                token_prob = probs[draft_tokens[i]].item()
                    
                eps = 1e-10
                logp_draft = math.log(token_prob + eps)
                log_ratio = logp_draft
                log_ratio = max(log_ratio, -10.0) / 10.0
                log_ratio = -log_ratio

                prob_features_list.append([
                    max_prob,
                    normalized_entropy,
                    log_ratio,
                ])

                top100_results = torch.topk(probs, k=100, dim=-1)
                top100_ids = top100_results.indices.tolist()

                draft_token_id = draft_tokens[i].item()

                if draft_token_id in top100_ids:
                    rank = top100_ids.index(draft_token_id) + 1
                    label = 1.0 - rank / 100.0
                else:
                    label = 0.0
                labels_list.append(label)

        prob_features = torch.tensor(
            prob_features_list, 
            dtype=torch.float32, 
            device=self.device,
        )
        
        labels = torch.tensor(
            labels_list, 
            dtype=torch.float32, 
            device=self.device
        )
        
        return prob_features, labels
    
    def collect_training_data(
        self,
        logits: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        draft_tokens: Union[List[int], torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        原有接口，保持兼容
        """
        if isinstance(draft_tokens, list):
            draft_tokens = torch.tensor(draft_tokens, device=self.device)
        
        return self.collect_training_data_single(logits, tree_attention_mask, draft_tokens)
        
    def _get_current_accepted_tokens_indices(
        self, 
        final_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
    ):
        seq_len = len(draft_tokens)
        prefix_len = attention_mask.shape[2] - seq_len

        logits = final_logits
        probs = torch.softmax(logits, dim=-1)
        is_leaf = torch.ones(seq_len, dtype=torch.bool)
        leaf_paths = []

        for i in range(seq_len - 1, -1, -1):
            if is_leaf[i]:
                path = [i]

                for j in range(i - 1, -1, -1):
                    if attention_mask[0, i, j + prefix_len] == 1:
                        is_leaf[j] = False
                        path.append(j)

                path.reverse()
                leaf_paths.append((i, path))

        best_path = None
        best_validated = -1

        for leaf_idx, path in leaf_paths:
            validated = 1
            for i in range(1, len(path)):
                idx = path[i]
                token_id = draft_tokens[idx].item()
                pred_id = probs[0, path[i - 1]].argmax().item()

                if pred_id == token_id:
                    validated += 1
                else:
                    break

            if validated > best_validated:
                best_validated = validated
                best_path = path[:validated]
                
        last_index = best_path[-1]

        return best_path, best_validated
    
    def train_step(
        self,
        middle_hidden_states: torch.Tensor,
        final_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> dict:
        accepted_indices, best_validated = self._get_current_accepted_tokens_indices(final_logits, attention_mask, draft_tokens)
        logger.info(f"train_step, accepted_indices: {accepted_indices}")
        prob_features, labels = self.collect_training_data(
            middle_hidden_states,                
            attention_mask,
            draft_tokens)

        self.ite = self.ite + 1
        
        self.decision_net.train()
        self.optimizer.zero_grad()
        
        tree_size = draft_tokens.shape[0]
        
        with torch.enable_grad():
            predictions, quality_scores = self.decision_net(prob_features)

        pos_count = labels.sum()
        neg_count = tree_size - pos_count
        
        if pos_count > 0:
            pos_weight = neg_count / pos_count
            sample_weights = torch.where(labels == 1, pos_weight, 1.0)
        else:
            sample_weights = torch.ones_like(labels)
            
        logger.info(f"predictions: {predictions}")
        logger.info(f"labels: {labels}")
        
        bce_loss = F.binary_cross_entropy(
            predictions, 
            labels, 
            weight=sample_weights
        )
        
        logger.info(f"train_step, bce_loss: {bce_loss}")
        total_loss = bce_loss
        total_loss.backward()
        self.optimizer.step()
        self.decision_net.eval()
        
        return {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'avg_quality_score': quality_scores.mean().item(),
            'avg_threshold': 0,
            'pos_count': pos_count.item(),
            'neg_count': neg_count.item()
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'decision_net': self.decision_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ite': self.ite,
            'g_ite': self.g_ite,
        }, path)
        self.g_ite = self.g_ite + 1
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.decision_net.load_state_dict(checkpoint['decision_net'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'ite' in checkpoint:
            self.ite = checkpoint['ite']
        if 'g_ite' in checkpoint:
            self.g_ite = checkpoint['g_ite']
    
    def get_network_condition(self):
        """Override this in subclass if needed"""
        return NetworkCondition.mock()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get pruning metrics"""
        prune_rate = 0
        accuracy = 0
        
        return {
            'prune_rate': prune_rate,
            'accuracy': accuracy,
        }