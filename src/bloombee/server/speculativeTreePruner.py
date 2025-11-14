import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)


class PruningMethod(Enum):
    """Enumeration of available pruning methods"""
    SIMPLE_PROBABILITY = "simple_probability"
    ADAPTIVE_NEURAL = "adaptive_neural"


@dataclass
class BranchContext:
    """Context information for a branch in the speculation tree"""
    tree_depth: int
    num_siblings: int
    parent_acceptance_prob: float
    branch_index: int
    token_id: int  # Added for tracking
    position: int  # Position in sequence


# @dataclass
# class NetworkCondition:
#     """Network statistics for adaptive pruning"""
#     avg_acceptance_rate: float
#     recent_latency: float
#     tokens_generated: int
#     pruning_accuracy: float
#     bandwidth_usage: float = 0.8  # Network bandwidth utilization
#     gpu_memory_usage: float = 0.7  # GPU memory utilization


@dataclass
class PruningConfig:
    """Configuration for pruning strategies"""
    method: PruningMethod = PruningMethod.SIMPLE_PROBABILITY
    
    # Simple probability method config
    simple_threshold: float = 0.3
    simple_top_k: int = 5
    simple_use_entropy: bool = False
    
    # Adaptive neural method config
    neural_threshold: float = 0.5
    neural_hidden_size: int = 128
    neural_dropout: float = 0.1
    neural_learning_rate: float = 1e-4
    neural_speedup_weight: float = 0.1
    
    # Shared config
    max_branches: int = 32
    min_keep_branches: int = 5
    enable_caching: bool = True
    cache_size: int = 1000


class PrunerInterface(ABC):
    """Abstract base class for all pruning strategies"""
    
    @abstractmethod
    def prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
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
    def update_statistics(self, feedback: Dict[str, Any]):
        """Update internal statistics based on pruning outcomes"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        pass

class SimpleProbabilityPruner(PrunerInterface):
    """
    Simple probability-based pruner using middle layer LM head
    Prunes branches based on token probability from intermediate layer
    Works with depth-first ordered sequences and attention mask
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        config: PruningConfig,
        tie_weights: Optional[nn.Linear] = None
    ):
        self.config = config
        self.vocab_size = vocab_size
        
        # Middle layer LM head
        if tie_weights is not None:
            self.lm_head = tie_weights
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.lm_head.requires_grad_(False)
            self.lm_head.to(dtype=torch.float16)
        
        # Statistics
        self.total_branches = 0
        self.pruned_branches = 0
        self.correct_prunes = 0
        
        # Cache for repeated computations
        self.cache = {} if config.enable_caching else None
    
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
        
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        # Get middle layer logits and probabilities
        logger.info(f"draft_tokens: {draft_tokens}")
        logger.info(f"middle_hidden_states: {middle_hidden_states}")
        middle_logits = self.lm_head(middle_hidden_states)
        logger.info(f"middle_logits: {middle_logits}")
        probs = F.softmax(middle_logits, dim=-1)
        logger.info(f"probs: {probs}")
        logger.info(f"seq_len: {seq_len}")
        logger.info(f"tree_attention_mask : {tree_attention_mask.shape}, {tree_attention_mask}")
        
        # Initialize keep mask (all True initially)
        keep_mask = torch.ones(seq_len, dtype=torch.bool)
        
        # Track which nodes are discarded (for skipping descendants)
        discarded = torch.zeros(seq_len, dtype=torch.bool)
        
        # Store scores for all nodes (for statistics and fallback)
        scores = torch.zeros(seq_len)
        
        # Process each node in depth-first order
        for i in range(seq_len):
            # Skip if already discarded by ancestor
            if discarded[i]:
                keep_mask[i] = False
                scores[i] = 0.0  # Set score to 0 for discarded nodes
                continue
            
            # logger.info(f"draft_tokens[i]: {draft_tokens[i]}")
            
            # Get token probability
            token_prob = probs[0, i, draft_tokens[i]]
            
            # logger.info(f"probs shape : {probs.shape}, tree_attention_mask shape: {tree_attention_mask.shape}")
            score = token_prob
            
            scores[i] = score
            
            # Check if score meets threshold
            if score < self.config.simple_threshold:
                keep_mask[i] = False
                discarded[i] = True
                
                # Mark all descendants as discarded
                # Descendants are nodes j > i where j can attend to i
                for j in range(i + 1, seq_len):
                    if tree_attention_mask[0, j, i + prefix_len] == 1:
                        discarded[j] = True
                        keep_mask[j] = False
        
        # Ensure minimum branches are kept
        kept_count = keep_mask.sum().item()
        logger.info(f"prune_branches keep_mask: {keep_mask}")
        logger.info(f"prune_branches discarded: {discarded}")
        logger.info(f"prune_branches kept_count: {kept_count}")
        # logger.info(f"prune_branches tree_attention_mask: {tree_attention_mask}")
        if kept_count < self.config.min_keep_branches:
            # Initialize all nodes as potential leaf nodes
            is_leaf = torch.ones(seq_len, dtype=torch.bool)
            
            # Store leaf nodes with their path scores
            leaf_paths = []  # List of (score, leaf_idx, path_indices)
            
            # Traverse from back to front
            for i in range(seq_len - 1, -1, -1):
                if is_leaf[i]:
                    # This is a leaf node, calculate its path score
                    path_indices = [i]
                    path_score = scores[i].item()
                    
                    # Find all ancestors of node i (from i-1 to 0)
                    for j in range(i - 1, -1, -1):
                        # Check if j is an ancestor of i
                        if tree_attention_mask[0, i, j + prefix_len] == 1:
                            # Mark j as non-leaf (it has descendants)
                            is_leaf[j] = False
                            
                            # Add to path
                            path_indices.append(j)
                            # Multiply path score
                            path_score *= scores[j].item()
                    
                    # Save this leaf and its path score
                    path_indices.reverse()  # Order from root to leaf
                    leaf_paths.append((path_score, i, path_indices))
            
            # Sort leaf nodes by path score (high to low)
            logger.info(f"prune_branches is_leaf : {is_leaf}")
            leaf_paths.sort(key=lambda x: x[0], reverse=True)
            logger.info(f"prune_branches leaf_paths : {leaf_paths}")
            
            # Keep top branches (complete paths from root to leaf)
            # keep_mask = torch.zeros(seq_len, dtype=torch.bool)
            branches_kept = 0
            
            for path_score, leaf_idx, path_indices in leaf_paths:
                if branches_kept >= self.config.min_keep_branches:
                    break
                
                # Keep all nodes in this path
                for idx in path_indices:
                    keep_mask[idx] = True
                
                branches_kept += 1
        
        # # Limit maximum branches
        # kept_count = keep_mask.sum().item()
        # if kept_count > self.config.max_branches:
        #     # Keep only top-scoring nodes
        #     scores_masked = scores * keep_mask.float()
        #     _, top_indices = torch.topk(scores_masked, self.config.max_branches)
        #     new_keep_mask = torch.zeros_like(keep_mask)
            
        #     # Keep top nodes and their ancestors
        #     for idx in top_indices:
        #         new_keep_mask[idx] = True
        #         # Keep all ancestors
        #         for j in range(idx):
        #             if tree_attention_mask[idx, j] == 1:
        #                 new_keep_mask[j] = True
            
        #     keep_mask = new_keep_mask
        
        # Get final indices
        keep_indices = torch.where(keep_mask)[0].tolist()
        prune_indices = torch.where(~keep_mask)[0].tolist()
        
        # Update statistics
        self.total_branches += seq_len
        self.pruned_branches += len(prune_indices)
        
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'keep_probs': scores.tolist(),
            'keep_mask': keep_mask,
            'metadata': {
                'middle_logits': middle_logits,
                'threshold_used': self.config.simple_threshold,
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
    
    def _get_depth_distribution(
        self,
        attention_mask: torch.Tensor,
        keep_mask: torch.Tensor
    ) -> Dict[int, int]:
        """Get distribution of kept nodes by depth"""
        depth_dist = {}
        for i in range(len(keep_mask)):
            if keep_mask[i]:
                depth = torch.sum(attention_mask[i, :i]).item()
                depth_dist[depth] = depth_dist.get(depth, 0) + 1
        return depth_dist
    
    def update_statistics(self, feedback: Dict[str, Any]):
        """Update pruning statistics"""
        if 'correct_prunes' in feedback:
            self.correct_prunes += feedback['correct_prunes']
    
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

@dataclass
class NetworkCondition:
    """Network condition metrics"""
    latency: float  # ms
    bandwidth: float  # Mbps
    packet_loss: float  # 0-1
    throughput: float  # Current throughput Mbps
    
    def to_features(self) -> List[float]:
        """Convert to normalized features for neural network"""
        return [
            min(self.latency / 200.0, 1.0),  # Normalize to [0, 1]
            min(self.bandwidth / 100.0, 1.0),
            self.packet_loss,
            min(self.throughput / 100.0, 1.0)
        ]
    
    @classmethod
    def mock(cls, condition_type: str = 'normal'):
        """Mock network condition for training"""
        conditions = {
            'good': {'latency': 10, 'bandwidth': 100, 'packet_loss': 0.001, 'throughput': 90},
            'normal': {'latency': 50, 'bandwidth': 50, 'packet_loss': 0.01, 'throughput': 40},
            'poor': {'latency': 150, 'bandwidth': 20, 'packet_loss': 0.05, 'throughput': 15}
        }
        
        base = conditions.get(condition_type, conditions['normal'])
        # Add some randomness
        return cls(
            latency=base['latency'] * (1 + random.uniform(-0.2, 0.2)),
            bandwidth=base['bandwidth'] * (1 + random.uniform(-0.1, 0.1)),
            packet_loss=base['packet_loss'] * (1 + random.uniform(-0.3, 0.3)),
            throughput=base['throughput'] * (1 + random.uniform(-0.15, 0.15))
        )

class DualPathPruner(nn.Module):
    """
    两条路径的网络：
    - 质量路径：评估token质量（不看网络）
    - 阈值路径：根据网络状况动态调整决策阈值
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # 质量评估路径（只看token特征，不看网络）
        self.quality_path = nn.Sequential(
            nn.Linear(4, hidden_size),  # prob(3) + acceptance(1)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # 输出质量分数
        )
        
        # 阈值调整路径（只看网络状况）
        self.threshold_path = nn.Sequential(
            nn.Linear(4, hidden_size // 2),  # network features(4)
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # 输出阈值调整量
        )
    
    def forward(self, prob_features, network_features):
        """
        Args:
            prob_features: [batch, 4] - max_prob, entropy, token_prob, acceptance_rate
            network_features: [batch, 4] - bandwidth, latency, packet_loss, jitter
        
        Returns:
            decision_prob: [batch] - 最终决策概率
            quality_score: [batch] - 质量分数
            threshold_adjust: [batch] - 阈值调整量
        """
        quality_score = self.quality_path(prob_features).squeeze(-1)  # [batch]
        threshold_adjust = self.threshold_path(network_features).squeeze(-1)  # [batch]
        
        # 决策分数 = 质量 - 阈值
        # 网络好时，threshold小，容易keep
        # 网络差时，threshold大，难以keep
        decision_score = quality_score - threshold_adjust
        
        # 通过sigmoid转换为概率
        decision_prob = torch.sigmoid(decision_score)
        
        return decision_prob, quality_score, threshold_adjust


class AdaptiveNeuralPruner:
    """
    Neural pruner with dual-path architecture
    """
    
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
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        self.lm_head.requires_grad_(False)
        self.lm_head.to(dtype=torch.float16)
        
        # Dual-path decision network
        self.decision_net = DualPathPruner(hidden_size=neural_hidden).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.decision_net.parameters(), 
            lr=1e-4
        )
        
        # Historical acceptance rate tracking
        self.acceptance_history = deque(maxlen=100)
        self.current_acceptance_rate = 0.5
        
        # Training mode flag
        self.training = False
    
    def extract_features(
        self,
        logits: torch.Tensor,
        parent_position: int,
        network_condition: NetworkCondition,
        acceptance_rate: float,
        draft_token: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Extract features and split into prob_features and network_features
        TODO: position error
        Returns:
            prob_features: [4] - max_prob, entropy, token_prob, acceptance_rate
            network_features: [4] - bandwidth, latency, packet_loss, jitter
        """
        
        # 获取特定位置的logits
        logits_at_pos = logits[0, parent_position]  # [vocab_size]
        
        # 计算概率
        probs = F.softmax(logits_at_pos, dim=-1)
        
        # 概率特征
        max_prob = torch.max(probs).item()
        logger.info(f"max_prob: {max_prob}")
        entropy = -torch.sum(torch.where(
            probs > 1e-10,
            probs * torch.log(probs),
            torch.zeros_like(probs)
        ))
        logger.info(f"entropy: {entropy}")
        log_vocab_size = torch.log(torch.tensor(float(self.vocab_size)))
        logger.info(f"log_vocab_size: {log_vocab_size}")
        if log_vocab_size > 0:
            normalized_entropy = (entropy / log_vocab_size).item()
        else:
            normalized_entropy = 0.0  # vocab_size <= 1 时
        logger.info(f"normalized_entropy: {normalized_entropy}")
        
        # Token概率
        if draft_token is not None:
            token_prob = probs[draft_token].item()
        else:
            token_prob = torch.topk(probs, k=min(5, self.vocab_size)).values.sum().item()
        
        # 组合特征
        prob_features = torch.tensor(
            [max_prob, normalized_entropy, token_prob, acceptance_rate],
            dtype=torch.float32,
            device=self.device
        )
        
        network_features = torch.tensor(
            network_condition.to_features(),  # [bandwidth, latency, packet_loss, jitter]
            dtype=torch.float32,
            device=self.device
        )
        
        return prob_features, network_features
    
    def _get_parent_postion(self, i, mask):
        for j in range(i-1, -1, -1):
            if mask[0, i, j] == 1:
                return j
        return i
    
    def prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: Optional[List[int]] = None,
        tree_attention_mask: torch.Tensor = None,
        network_condition = None,
    ) -> Dict:
        """
        Main pruning interface
        
        Args:
            middle_hidden_states: [1, seq_len, hidden_size]
            tree_attention_mask: [1, seq_len, total_len]
            network_condition: Current network state
            draft_tokens: Optional draft token IDs
        """
        
        seq_len = middle_hidden_states.shape[1]
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        # Mock network condition if not provided
        if network_condition is None:
            network_condition = self.get_network_condition() or NetworkCondition.mock()
        
        # Get logits from middle layer
        logits = self.lm_head(middle_hidden_states)
        
        # Initialize masks
        keep_mask = torch.ones(seq_len, dtype=torch.bool)
        discarded = torch.zeros(seq_len, dtype=torch.bool)
        decision_probs = torch.zeros(seq_len)
        quality_scores = torch.zeros(seq_len)
        threshold_adjusts = torch.zeros(seq_len)
        
        logger.info(f"prune_branches, seq_len: {seq_len}")
        logger.info(f"prune_branches, discarded: {discarded}")
        
        # Process each position
        for i in range(seq_len):
            if i == 0:
                keep_mask[0] = True
                decision_probs[0] = 1.0
                continue
            
            if discarded[i]:
                keep_mask[i] = False
                decision_probs[i] = 0.0
                continue
            
            # Extract features
            parent_postion = self._get_parent_postion(i, tree_attention_mask)
            prob_features, network_features = self.extract_features(
                logits, parent_postion,
                network_condition,
                self.current_acceptance_rate,
                draft_tokens[i] if draft_tokens is not None else None
            )
            
            logger.info(f"prune_branches, prob_features: {prob_features}")
            logger.info(f"prune_branches, network_features: {network_features}")
            
            # Get decision from dual-path network
            with torch.no_grad():
                prob, quality, threshold = self.decision_net(
                    prob_features.unsqueeze(0),
                    network_features.unsqueeze(0)
                )
                
                decision_probs[i] = prob.item()
                quality_scores[i] = quality.item()
                threshold_adjusts[i] = threshold.item()
                
                # Decision: >0.5 means keep
                keep = prob.item() > self.config.neural_threshold
                
                logger.info(f"prune_branches, prob: {prob}, quality: {quality}, threshold: {threshold}, i: {i}")
            
            if not keep:
                keep_mask[i] = False
                discarded[i] = True
                
                # Discard descendants
                for j in range(i + 1, seq_len):
                    if tree_attention_mask[0, j, i + prefix_len] == 1:
                        discarded[j] = True
                        keep_mask[j] = False
        
        # Ensure at least one branch is kept
        kept_count = keep_mask.sum().item()
        
        # logger.info(f"kept_count: {kept_count}")
        
        if kept_count < self.config.min_keep_branches:
            # Keep top-scoring branches
            is_leaf = torch.ones(seq_len, dtype=torch.bool)
            leaf_paths = []
            
            for i in range(seq_len - 1, -1, -1):
                if is_leaf[i]:
                    path_indices = [i]
                    path_score = decision_probs[i].item()
                    
                    for j in range(i - 1, -1, -1):
                        if tree_attention_mask[0, i, j + prefix_len] == 1:
                            is_leaf[j] = False
                            path_indices.append(j)
                            path_score *= decision_probs[j].item()
                    
                    path_indices.reverse()
                    leaf_paths.append((path_score, i, path_indices))
            
            leaf_paths.sort(key=lambda x: x[0], reverse=True)
            branches_kept = 0
            
            for path_score, leaf_idx, path_indices in leaf_paths:
                if branches_kept >= self.config.min_keep_branches:
                    break
                for idx in path_indices:
                    keep_mask[idx] = True
                branches_kept += 1
        
        # Get final indices
        keep_indices = torch.where(keep_mask)[0].tolist()
        prune_indices = torch.where(~keep_mask)[0].tolist()
        
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'decision_probs': decision_probs.cpu().tolist(),
            'quality_scores': quality_scores.cpu().tolist(),
            'threshold_adjusts': threshold_adjusts.cpu().tolist(),
            'network_condition': network_condition,
            'acceptance_rate': self.current_acceptance_rate
        }
    
    def collect_training_data(
        self,
        middle_hidden_states: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        accepted_indices: List[int],
        network_condition,
        draft_tokens: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect features and labels for training
        
        Returns:
            prob_features: [seq_len, 4]
            network_features: [seq_len, 4]
            labels: [seq_len]
        """
        
        seq_len = middle_hidden_states.shape[0]
        middle_hidden_states = middle_hidden_states.to(self.device)
        
        # Get logits
        logits = self.lm_head(middle_hidden_states.unsqueeze(0))
        
        prob_features_list = []
        network_features_list = []
        labels_list = []
        
        for i in range(seq_len):
            # Extract features
            prob_feat, net_feat = self.extract_features(
                logits, i,
                network_condition,
                self.current_acceptance_rate,
                draft_tokens[i] if draft_tokens else None
            )
            prob_features_list.append(prob_feat)
            network_features_list.append(net_feat)
            
            # Label: 1 if accepted, 0 if rejected
            label = 1.0 if i in accepted_indices else 0.0
            labels_list.append(label)
        
        return (
            torch.stack(prob_features_list),
            torch.stack(network_features_list),
            torch.tensor(labels_list, device=self.device)
        )
    
    def train_step(
        self,
        prob_features: torch.Tensor,      # [tree_size, 4]
        network_features: torch.Tensor,   # [tree_size, 4]
        labels: torch.Tensor,             # [tree_size]
        middle_hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
        acceptance_rate,
        alpha: float = 1.0,               # BCE loss weight
        beta: float = 0.1,                 # Pruning alignment weight
    ) -> dict:
        """
        Single training step for ONE tree
        
        Args:
            prob_features: Token probability features
            network_features: Network condition features
            labels: Whether each token was accepted
            alpha: Weight for BCE loss
            beta: Weight for pruning alignment loss
        """
        
        self.decision_net.train()
        self.optimizer.zero_grad()
        
        tree_size = prob_features.shape[0]
        
        # Forward pass
        predictions, quality_scores, threshold_adjusts = self.decision_net(
            prob_features, 
            network_features
        )
        
        # === Loss 1: Token Quality Prediction (with class weighting) ===
        pos_count = labels.sum()
        neg_count = tree_size - pos_count
        
        if pos_count > 0:
            pos_weight = neg_count / pos_count
            # 为正样本加权
            sample_weights = torch.where(labels == 1, pos_weight, 1.0)
        else:
            sample_weights = torch.ones_like(labels)
        
        bce_loss = F.binary_cross_entropy(
            predictions, 
            labels, 
            weight=sample_weights
        )
        
        # === Loss 2: Network-Aware Pruning Alignment ===
        # 当前预测的pruning rate
        current_pruning_rate = (predictions < 0.5).float().mean()
        
        # 从网络特征计算目标pruning rate
        # 使用第一个token的网络特征（整棵树共享）
        bandwidth = network_features[0, 0].item()
        network_severity = 1.0 - min(bandwidth / 100.0, 1.0)
        target_pruning_rate = 0.2 + 0.6 * network_severity  # 20%-80%
        
        # Alignment loss
        pruning_alignment = (current_pruning_rate - target_pruning_rate) ** 2
        
        # === Total Loss ===
        total_loss = alpha * bce_loss + beta * pruning_alignment
        
        total_loss.backward()
        self.optimizer.step()
        
        self.decision_net.eval()
        
        return {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'pruning_alignment': pruning_alignment.item(),
            'current_pruning_rate': current_pruning_rate.item(),
            'target_pruning_rate': target_pruning_rate,
            'network_severity': network_severity,
            'avg_quality_score': quality_scores.mean().item(),
            'avg_threshold': threshold_adjusts.mean().item(),
            'pos_count': pos_count.item(),
            'neg_count': neg_count.item()
        }
    
    def update_acceptance_rate(self, rate: float):
        """Update historical acceptance rate"""
        self.acceptance_history.append(rate)
        self.current_acceptance_rate = sum(self.acceptance_history) / len(self.acceptance_history)
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'decision_net': self.decision_net.state_dict(),
            'lm_head': self.lm_head.state_dict(),
            'acceptance_history': list(self.acceptance_history),
            'current_acceptance_rate': self.current_acceptance_rate
        }, path)
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.decision_net.load_state_dict(checkpoint['decision_net'])
        self.lm_head.load_state_dict(checkpoint['lm_head'])
        self.acceptance_history = deque(checkpoint['acceptance_history'], maxlen=100)
        self.current_acceptance_rate = checkpoint['current_acceptance_rate']
    
    def get_network_condition(self):
        """Override this in subclass if needed"""
        return None


class BloombeePrunerFactory:
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
            logger.info("Creating Simple Probability Pruner")
            return SimpleProbabilityPruner(hidden_size, vocab_size, config, tie_weights)
        elif method == PruningMethod.ADAPTIVE_NEURAL:
            logger.info("Creating Adaptive Neural Pruner")
            return AdaptiveNeuralPruner(hidden_size, vocab_size, 64, 'cuda', config)
        else:
            raise ValueError(f"Unknown pruning method: {method}")


class BloombeePrunerManager:
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
            middle_hidden_states,
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
    
    def _build_branch_contexts(
        self,
        draft_tokens: List[int],
        tree_structure: Dict[str, Any]
    ) -> List[BranchContext]:
        """Build branch contexts from tree structure"""
        contexts = []
        
        for i, token_id in enumerate(draft_tokens):
            context = BranchContext(
                tree_depth=tree_structure.get('depths', [1] * len(draft_tokens))[i],
                num_siblings=tree_structure.get('num_siblings', [len(draft_tokens)] * len(draft_tokens))[i],
                parent_acceptance_prob=tree_structure.get('parent_probs', [1.0] * len(draft_tokens))[i],
                branch_index=i,
                token_id=token_id,
                position=tree_structure.get('positions', list(range(len(draft_tokens))))[i]
            )
            contexts.append(context)
        
        return contexts
    
    def update_with_verification_results(
        self,
        kept_indices: List[int],
        accepted_indices: List[int],
        latency: float
    ):
        """Update pruner with verification results"""
        feedback = {
            'acceptance': len(accepted_indices) / max(len(kept_indices), 1),
            'latency': latency,
            'pruning_correct': len(set(kept_indices) & set(accepted_indices)) / max(len(kept_indices), 1),
            'branch_acceptances': [i in accepted_indices for i in kept_indices]
        }
        
        self.pruner.update_statistics(feedback)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from both manager and pruner"""
        return {
            'manager': {
                'total_tokens': self.total_tokens,
                'pruned_tokens': self.pruned_tokens,
                'speedup_factor': self.speedup_factor,
                'pruning_method': self.config.method.value
            },
            'pruner': self.pruner.get_metrics()
        }
    
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
        state = torch.load(path, map_location=self.device)
    
        # Restore manager metrics
        self.total_tokens = state['manager_metrics']['total_tokens']
        self.pruned_tokens = state['manager_metrics']['pruned_tokens']
        self.speedup_factor = state['manager_metrics']['speedup_factor']
        
        # Load pruner-specific state if available
        if 'pruner_path' in state and hasattr(self.pruner, 'load_model'):
            self.pruner.load_model(state['pruner_path'])
            logger.info(f"Loaded pruner model from {state['pruner_path']}")
        
        logger.info(f"Loaded pruner manager state from {path}")
        
    def train_model(self, middle_hidden_states, final_hidden_states, attention_mask, draft_tokens, acceptance_rate):
        if hasattr(self.pruner, 'train_step'):
            self.pruner.train_step(middle_hidden_states, final_hidden_states, attention_mask, draft_tokens, acceptance_rate)
        