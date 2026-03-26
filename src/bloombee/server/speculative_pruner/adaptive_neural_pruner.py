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
        
        self.lm_head = MidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to("cuda")
        lm_head_weights_path = hf_hub_download(
            repo_id="xxiong59/lm-head-for-speculative-pruning",
            filename="lm_head_llama30B-15.pt",
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
        
        self.optimizer = torch.optim.AdamW(
            self.decision_net.parameters(), 
            lr=1e-4
        )
        
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

        # Pre-compute log_vocab_size constant
        self._log_vocab_size = math.log(float(vocab_size)) if vocab_size > 1 else 1.0

    # -------------------------------------------------------------------------
    # Vectorized feature extraction (replaces collect_training_data_single loop)
    # -------------------------------------------------------------------------
    def _compute_prob_features_batched(
        self,
        logits: torch.Tensor,           # [B, seq_len, vocab_size]  float16 or float32
        tree_attention_mask: torch.Tensor,  # [B, seq_len, total_len]  bool/uint8
        draft_tokens: torch.Tensor,     # [B, seq_len]
    ) -> torch.Tensor:
        """
        Fully vectorized replacement for the seq_len Python loop in
        collect_training_data_single.

        Returns prob_features: [B, seq_len-1, 3]  (positions 1..seq_len-1)
        """
        B, S, V = logits.shape
        prefix_len = tree_attention_mask.shape[2] - S
        device = logits.device

        # ── 1. parent indices for each (b, i) where i in [1, S) ────────────
        # mask slice we care about: [B, S, S]  (child × parent)
        child_to_parent_mask = tree_attention_mask[:, :, prefix_len:].to(device)  # [B, S, S]

        # For each child i (rows 1..S-1), find the *last* True column j < i.
        # Strategy: multiply col indices by mask, take max over j.
        # We only look at j < i by zeroing the upper triangle.
        col_idx = torch.arange(S, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,S]
        # Zero out j >= i (upper triangle + diagonal)
        causal_mask = (col_idx < torch.arange(S, device=device).view(1, S, 1))  # [1,S,S]
        masked_cols = (child_to_parent_mask & causal_mask).long() * (col_idx + 1)  # +1 so 0=no parent
        parent_indices = masked_cols.max(dim=-1).values - 1  # [B, S], -1 means no parent found
        # Clamp: if somehow -1 stay 0 (only happens for root, which we skip)
        parent_indices = parent_indices.clamp(min=0)  # [B, S]

        # ── 2. Gather logits at parent positions ────────────────────────────
        # We need logits[b, parent_indices[b, i], :] for i in 1..S-1
        # parent_indices: [B, S] → expand for gather
        gather_idx = parent_indices[:, 1:].unsqueeze(-1).expand(B, S - 1, V)  # [B, S-1, V]
        # Cast logits to float32 for numerical stability
        logits_f32 = logits.float()
        parent_logits = logits_f32.gather(1, gather_idx)  # [B, S-1, V]

        # ── 3. Softmax over vocab ────────────────────────────────────────────
        probs = torch.softmax(parent_logits, dim=-1)  # [B, S-1, V]

        # ── 4. max_prob ──────────────────────────────────────────────────────
        max_prob = probs.max(dim=-1).values  # [B, S-1]

        # ── 5. normalized entropy ────────────────────────────────────────────
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, S-1]
        normalized_entropy = entropy / self._log_vocab_size  # [B, S-1]

        # ── 6. token_prob for draft token at position i ──────────────────────
        token_ids = draft_tokens[:, 1:].long().to(device)  # [B, S-1]
        token_prob = probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

        # ── 7. log_ratio feature ─────────────────────────────────────────────
        log_ratio = torch.log(token_prob.clamp(min=1e-10))   # [B, S-1]
        log_ratio = log_ratio.clamp(min=-10.0) / 10.0
        log_ratio = -log_ratio

        # ── 8. Stack features ────────────────────────────────────────────────
        prob_features = torch.stack([max_prob, normalized_entropy, log_ratio], dim=-1)  # [B, S-1, 3]

        return prob_features

    # -------------------------------------------------------------------------
    # Vectorized label computation (used in train_step)
    # -------------------------------------------------------------------------
    def _compute_labels_batched(
        self,
        probs: torch.Tensor,        # [B, S-1, V]
        draft_tokens: torch.Tensor, # [B, S]
        top_k: int = 100,
    ) -> torch.Tensor:
        """Returns labels [B, S-1]."""
        B, Sm1, V = probs.shape
        token_ids = draft_tokens[:, 1:].long()  # [B, S-1]

        # topk ranks
        _, topk_ids = probs.topk(top_k, dim=-1)  # [B, S-1, k]

        # Check if draft token is in top-k and find rank
        match = (topk_ids == token_ids.unsqueeze(-1))  # [B, S-1, k]
        in_topk = match.any(dim=-1)  # [B, S-1]
        # rank = first True position + 1
        rank = match.float().argmax(dim=-1) + 1  # [B, S-1]  (0 if not found, but we mask)

        labels = torch.where(in_topk, 1.0 - rank.float() / top_k, torch.zeros_like(rank.float()))
        return labels

    # -------------------------------------------------------------------------
    # Vectorized descendent-pruning  — fully batched, no Python loop over B
    # -------------------------------------------------------------------------
    @staticmethod
    def _propagate_prune_mask_batched(
        initial_keep: torch.Tensor,         # [B, S]  bool
        tree_attention_mask: torch.Tensor,  # [B, S, total_len]
        prefix_len: int,
    ) -> torch.Tensor:
        """
        Propagate 'discard' downward through the tree for the entire batch at once.

        Key insight: the tree is topologically ordered (parent index < child index),
        so a single left-to-right pass over S suffices — no fixed-point iteration.
        Each step is a tensor op over [B, S], eliminating the inner batch loop.

        Returns final_keep: [B, S] bool
        """
        device = initial_keep.device

        # adj[b, i, j] = True: token i attends to token j (j is ancestor of i)
        adj = tree_attention_mask[:, :, prefix_len:].to(device)  # [B, S, S]

        discarded = ~initial_keep  # [B, S]

        # O(S) iterations, each a [B, S] tensor op — no inner Python loop over B
        for i in range(1, initial_keep.shape[1]):
            # adj[:, i, :] — [B, S]: which tokens are parents of position i
            has_discarded_parent = (adj[:, i, :] & discarded).any(dim=-1)  # [B]
            discarded[:, i] = discarded[:, i] | has_discarded_parent

        return ~discarded  # [B, S]

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------
    def prune_branches(
        self,
        norm_hidden_states: torch.Tensor,
        draft_tokens: Union[List[int], torch.Tensor] = None,
        tree_attention_mask: torch.Tensor = None,
        network_condition=None,
    ) -> Dict:
        # ── normalise dims ────────────────────────────────────────────────────
        if norm_hidden_states.dim() == 2:
            norm_hidden_states = norm_hidden_states.unsqueeze(0)
            if tree_attention_mask.dim() == 2:
                tree_attention_mask = tree_attention_mask.unsqueeze(0)
            if isinstance(draft_tokens, list):
                draft_tokens = [draft_tokens]
            elif isinstance(draft_tokens, torch.Tensor) and draft_tokens.dim() == 1:
                draft_tokens = draft_tokens.unsqueeze(0)

        B, S, _ = norm_hidden_states.shape
        device = norm_hidden_states.device
        prefix_len = tree_attention_mask.shape[2] - S

        if network_condition is None:
            network_condition = self.get_network_condition() or NetworkCondition.mock()

        # ── 1. LM head → logits  [B, S, V] ───────────────────────────────────
        logits = self.lm_head(norm_hidden_states)  # float16

        # ── 2. Normalise draft_tokens ─────────────────────────────────────────
        if isinstance(draft_tokens, list):
            if isinstance(draft_tokens[0], list):
                draft_tokens = torch.tensor(draft_tokens, device=device)
            else:
                draft_tokens = torch.tensor(draft_tokens, device=device).unsqueeze(0)

        # ── 3. Compute all features at once  [B, S-1, 3] ─────────────────────
        with torch.no_grad():
            prob_features = self._compute_prob_features_batched(
                logits, tree_attention_mask, draft_tokens
            )  # [B, S-1, 3]

            # ── 4. Run decision_net on entire batch×seq in one forward pass ──
            # Flatten to [B*(S-1), 3], run, reshape back
            flat_features = prob_features.view(B * (S - 1), 3)
            flat_probs, flat_quality = self.decision_net(flat_features)
            decision_probs_inner = flat_probs.view(B, S - 1)    # [B, S-1]
            quality_scores_inner = flat_quality.view(B, S - 1)  # [B, S-1]

        # ── 5. Build per-token keep decisions ────────────────────────────────
        # decision_probs for position 0 is always 1.0
        ones_col = torch.ones(B, 1, device=device)
        decision_probs = torch.cat([ones_col, decision_probs_inner], dim=1)   # [B, S]
        quality_scores = torch.cat([torch.zeros(B, 1, device=device), quality_scores_inner], dim=1)

        initial_keep = (decision_probs > self.config.neural_threshold)  # [B, S]
        initial_keep[:, 0] = True  # root always kept

        # ── 6. Propagate discards down the tree — fully batched ───────────────
        final_keep = self._propagate_prune_mask_batched(
            initial_keep, tree_attention_mask, prefix_len
        )  # [B, S]

        # ── 7. Build padded keep/prune index tensors — no Python loops ────────
        # Assign a dense rank to each kept position within each batch row.
        # Positions that are NOT kept get rank S (out-of-range sentinel).
        keep_rank = torch.where(final_keep,
                                torch.arange(S, device=device).unsqueeze(0).expand(B, S),
                                torch.full((B, S), S, device=device))  # [B, S]
        sorted_keep = keep_rank.sort(dim=-1).values           # [B, S]

        # Number of kept tokens per batch item
        valid_lengths = final_keep.sum(dim=-1)                # [B]
        max_keep = int(valid_lengths.max().item())

        # keep indices: take first max_keep columns, replace sentinel S with -1
        padded_keep = sorted_keep[:, :max_keep].clone()
        padded_keep[padded_keep == S] = -1                    # [B, max_keep]

        # prune indices: inverse of keep
        prune_mask = ~final_keep
        prune_rank = torch.where(prune_mask,
                                 torch.arange(S, device=device).unsqueeze(0).expand(B, S),
                                 torch.full((B, S), S, device=device))
        sorted_prune = prune_rank.sort(dim=-1).values
        num_pruned = prune_mask.sum(dim=-1)
        max_prune = int(num_pruned.max().item()) if num_pruned.any() else 0
        if max_prune > 0:
            padded_prune = sorted_prune[:, :max_prune].clone()
            padded_prune[padded_prune == S] = -1              # [B, max_prune]
        else:
            padded_prune = torch.empty((B, 0), dtype=torch.long, device=device)

        return {
            'keep_indices': padded_keep,
            'prune_indices': padded_prune,
            'decision_probs': decision_probs,
            'quality_scores': quality_scores,
            'threshold_adjusts': 0,
            'network_condition': network_condition,
            'valid_lengths': valid_lengths,
        }

    # -------------------------------------------------------------------------
    # collect_training_data  (kept for train_step compatibility)
    # -------------------------------------------------------------------------
    def collect_training_data_single(
        self,
        logits: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-batch wrapper around the vectorized implementation."""
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        if tree_attention_mask.dim() == 2:
            tree_attention_mask = tree_attention_mask.unsqueeze(0)
        if draft_tokens.dim() == 1:
            draft_tokens = draft_tokens.unsqueeze(0)

        prob_features = self._compute_prob_features_batched(
            logits.float(), tree_attention_mask, draft_tokens
        )  # [1, S-1, 3]

        # Compute labels using vectorized method
        probs = torch.softmax(
            self._gather_parent_logits(logits.float(), tree_attention_mask, draft_tokens),
            dim=-1
        )  # [1, S-1, V]
        labels = self._compute_labels_batched(probs, draft_tokens)  # [1, S-1]

        return prob_features.squeeze(0), labels.squeeze(0)

    def _gather_parent_logits(
        self,
        logits: torch.Tensor,           # [B, S, V]
        tree_attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, S-1, V] logits at parent positions."""
        B, S, V = logits.shape
        prefix_len = tree_attention_mask.shape[2] - S
        device = logits.device

        col_idx = torch.arange(S, device=device).view(1, 1, S)
        causal_mask = col_idx < torch.arange(S, device=device).view(1, S, 1)
        child_to_parent = tree_attention_mask[:, :, prefix_len:]
        masked_cols = (child_to_parent & causal_mask).long() * (col_idx + 1)
        parent_indices = masked_cols.max(dim=-1).values - 1
        parent_indices = parent_indices.clamp(min=0)

        gather_idx = parent_indices[:, 1:].unsqueeze(-1).expand(B, S - 1, V)
        return logits.gather(1, gather_idx)

    def collect_training_data(
        self,
        logits: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        draft_tokens: Union[List[int], torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(draft_tokens, list):
            draft_tokens = torch.tensor(draft_tokens, device=self.device)
        return self.collect_training_data_single(logits, tree_attention_mask, draft_tokens)

    def _get_parent_position(self, i, mask, prefix, batch_idx=0):
        for j in range(i - 1, -1, -1):
            if mask[batch_idx, i, j + prefix] == True:
                return j
        return i

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
        accepted_indices, best_validated = self._get_current_accepted_tokens_indices(
            final_logits, attention_mask, draft_tokens
        )
        logger.info(f"train_step, accepted_indices: {accepted_indices}")
        prob_features, labels = self.collect_training_data(
            middle_hidden_states, attention_mask, draft_tokens
        )

        self.ite += 1
        self.decision_net.train()
        self.optimizer.zero_grad()

        tree_size = draft_tokens.shape[0]

        with torch.enable_grad():
            predictions, quality_scores = self.decision_net(prob_features)

        pos_count = labels.sum()
        neg_count = tree_size - pos_count

        if pos_count > 0:
            pos_weight = neg_count / pos_count
            sample_weights = torch.where(labels == 1, pos_weight, torch.ones_like(labels))
        else:
            sample_weights = torch.ones_like(labels)

        logger.info(f"predictions: {predictions}")
        logger.info(f"labels: {labels}")

        bce_loss = F.binary_cross_entropy(predictions, labels, weight=sample_weights)
        logger.info(f"train_step, bce_loss: {bce_loss}")

        bce_loss.backward()
        self.optimizer.step()
        self.decision_net.eval()

        return {
            'total_loss': bce_loss.item(),
            'bce_loss': bce_loss.item(),
            'avg_quality_score': quality_scores.mean().item(),
            'avg_threshold': 0,
            'pos_count': pos_count.item(),
            'neg_count': neg_count.item()
        }

    def save_model(self, path: str):
        torch.save({
            'decision_net': self.decision_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ite': self.ite,
            'g_ite': self.g_ite,
        }, path)
        self.g_ite += 1

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.decision_net.load_state_dict(checkpoint['decision_net'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'ite' in checkpoint:
            self.ite = checkpoint['ite']
        if 'g_ite' in checkpoint:
            self.g_ite = checkpoint['g_ite']

    def get_network_condition(self):
        return NetworkCondition.mock()

    def get_metrics(self) -> Dict[str, float]:
        return {'prune_rate': 0, 'accuracy': 0}