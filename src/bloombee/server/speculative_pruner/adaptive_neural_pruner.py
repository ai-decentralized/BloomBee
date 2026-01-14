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
            nn.Linear(3, hidden_size),  # prob(3) + acceptance(1)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.threshold_path = nn.Sequential(
            nn.Linear(4, hidden_size // 2),  # network features(4)
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, prob_features):
        quality_score = self.quality_path(prob_features).squeeze(-1)  # [batch]
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
        self.atc = 0 # accept_tokens_count
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

    def _get_parent_postion(self, i, mask, prefix):
        for j in range(i-1, -1, -1):
            if mask[0, i, j + prefix] == True:
                return j
        return i
    
    def prune_branches(
        self,
        norm_hidden_states: torch.Tensor,
        draft_tokens: Optional[List[int]] = None,
        tree_attention_mask: torch.Tensor = None,
        network_condition = None,
    ) -> Dict:
        seq_len = len(draft_tokens)
        prefix_len = tree_attention_mask.shape[2] - seq_len
        if network_condition is None:
            network_condition = self.get_network_condition() or NetworkCondition.mock()
            
        logits = self.lm_head(norm_hidden_states)
        
        # Initialize masks
        keep_mask = torch.ones(seq_len, dtype=torch.bool)
        discarded = torch.zeros(seq_len, dtype=torch.bool)
        decision_probs = torch.zeros(seq_len)
        quality_scores = torch.zeros(seq_len)
        threshold_adjusts = torch.zeros(seq_len)
        prob_features_list, _ = self.collect_training_data(
                logits, 
                tree_attention_mask,
                draft_tokens
            )
        for i in range(seq_len):
            if i == 0:
                keep_mask[0] = True
                decision_probs[0] = 1.0
                continue
            
            if discarded[i]:
                keep_mask[i] = False
                decision_probs[i] = 0.0
                continue
            
            prob_features = prob_features_list[i-1]
            with torch.no_grad():
                prob, quality = self.decision_net(prob_features.unsqueeze(0))
                
                decision_probs[i] = prob.item()
                quality_scores[i] = quality.item()
                threshold_adjusts[i] = 0
                keep = prob.item() > self.config.neural_threshold
                
            if not keep:
                keep_mask[i] = False
                discarded[i] = True
                
                # Discard descendants
                for j in range(i + 1, seq_len):
                    if tree_attention_mask[0, j, i + prefix_len] == 1:
                        discarded[j] = True
                        keep_mask[j] = False

        keep_indices = torch.where(keep_mask)[0].tolist()
        prune_indices = torch.where(~keep_mask)[0].tolist()
        
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'decision_probs': decision_probs.cpu().tolist(),
            'quality_scores': quality_scores.cpu().tolist(),
            'threshold_adjusts': 0,
            'network_condition': network_condition,
        }
    
    def collect_training_data(
        self,
        logits: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        draft_tokens: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        seq_len = len(draft_tokens)
        prefix_len = tree_attention_mask.shape[2] - seq_len

        with torch.no_grad():
            prob_features_list = []
            labels_list = []

            for i in range(1, seq_len):
                parent_postion = self._get_parent_postion(i, tree_attention_mask, prefix_len)
                
                logits_at_pos = logits[0, parent_postion]
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
                
                if draft_tokens is not None:
                    token_prob = probs[draft_tokens[i]].item()
                else:
                    token_prob = torch.topk(probs, k=min(5, self.vocab_size)).values.sum().item()
                    
                eps = 1e-10
                logp_draft = math.log(token_prob + eps)
                log_ratio  = logp_draft
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
        next_token = probs[0, last_index].argmax().item()
        logger.info(f"next token: {next_token}")

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
