import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
import logging

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


@dataclass
class NetworkCondition:
    """Network statistics for adaptive pruning"""
    avg_acceptance_rate: float
    recent_latency: float
    tokens_generated: int
    pruning_accuracy: float
    bandwidth_usage: float = 0.8  # Network bandwidth utilization
    gpu_memory_usage: float = 0.7  # GPU memory utilization


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
    min_keep_branches: int = 1
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

class AdaptiveNeuralPruner:
    """
    Neural pruner that directly outputs keep/prune decisions
    No threshold adjustment needed - network learns everything
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        neural_hidden: int = 64,
        device: str = 'cuda',
        config: PruningConfig = None,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.config = config
        
        # LM head for getting probabilities
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        
        # Neural network for direct decision
        # Input: 4 network + 3 probability + 1 acceptance = 8 features
        # Output: Single probability (>0.5 means keep, ≤0.5 means prune)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dt  = torch.float32
        torch.set_default_dtype(dt)
        self.decision_net = nn.Sequential(
            nn.Linear(8, neural_hidden),
            nn.LayerNorm(neural_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(neural_hidden, neural_hidden // 2),
            nn.LayerNorm(neural_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(neural_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.decision_net.parameters(), lr=1e-4)
        
        # Historical acceptance rate tracking
        self.acceptance_history = deque(maxlen=100)
        self.current_acceptance_rate = 0.5  # Initial guess
        
        # Training mode flag
        self.training = False

        
        
    def extract_features(
        self,
        logits: torch.Tensor,
        position: int,
        network_condition: NetworkCondition,
        acceptance_rate: float,
        draft_token: Optional[int] = None
    ) -> torch.Tensor:
        """Extract all features for decision making"""
        
        probs = F.softmax(logits[position], dim=-1)
        
        # Probability features
        max_prob = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        normalized_entropy = entropy / torch.log(torch.tensor(float(self.vocab_size)))
        
        if draft_token is not None:
            token_prob = probs[draft_token].item()
        else:
            # Use top-5 probability mass as proxy
            token_prob = torch.topk(probs, k=min(5, self.vocab_size)).values.sum().item()
        
        # Combine all features
        features = torch.tensor(
            network_condition.to_features() +  # 4 network features
            [max_prob, normalized_entropy, token_prob] +  # 3 probability features
            [acceptance_rate],  # 1 acceptance rate
            dtype=torch.float32,
            device=self.device
        )
        
        return features
    
    def prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        network_condition: Optional[NetworkCondition] = None,
        draft_tokens: Optional[List[int]] = None
    ) -> Dict:
        """
        Main pruning interface - directly outputs decisions
        
        Args:
            middle_hidden_states: [seq_len, hidden_size]
            tree_attention_mask: [seq_len, seq_len]
            network_condition: Current network state (will mock if None)
            draft_tokens: Optional draft token IDs
        """
        
        seq_len = middle_hidden_states.shape[0]
        
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        # Mock network condition if not provided
        if network_condition is None:
            network_condition = NetworkCondition.mock()
        
        # Move to device
        middle_hidden_states = middle_hidden_states.to(self.device)
        tree_attention_mask = tree_attention_mask.to(self.device)
        
        # Get logits from middle layer
        logits = self.lm_head(middle_hidden_states)
        
        # Initialize masks
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
        discarded = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        decision_probs = torch.zeros(seq_len, device=self.device)
        
        # Process each position in depth-first order
        for i in range(seq_len):
            if discarded[i]:
                keep_mask[i] = False
                decision_probs[i] = 0.0
                continue
            
            # Extract features
            features = self.extract_features(
                logits, i,
                network_condition,
                self.current_acceptance_rate,
                draft_tokens[i] if draft_tokens else None
            )
            
            # Get decision from neural network
            with torch.no_grad():
                prob = self.decision_net(features.unsqueeze(0)).squeeze()
                decision_probs[i] = prob
                
                # Direct decision: >0.5 means keep
                keep = prob > self.config.neural_threshold
            
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
                    path_score = decision_probs[i].item()
                    
                    # Find all ancestors of node i (from i-1 to 0)
                    for j in range(i - 1, -1, -1):
                        # Check if j is an ancestor of i
                        if tree_attention_mask[0, i, j + prefix_len] == 1:
                            # Mark j as non-leaf (it has descendants)
                            is_leaf[j] = False
                            
                            # Add to path
                            path_indices.append(j)
                            # Multiply path score
                            path_score *= decision_probs[j].item()
                    
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
        
        # Get final indices
        keep_indices = torch.where(keep_mask)[0].tolist()
        prune_indices = torch.where(~keep_mask)[0].tolist()
        
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'decision_probs': decision_probs.cpu().tolist(),
            'network_condition': network_condition,
            'acceptance_rate': self.current_acceptance_rate
        }
    
    def collect_training_data(
        self,
        middle_hidden_states: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        accepted_indices: List[int],
        network_condition: NetworkCondition,
        draft_tokens: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect features and labels for training
        
        Args:
            accepted_indices: Indices that were actually accepted by main model
            
        Returns:
            features: [seq_len, 8] tensor of features
            labels: [seq_len] tensor of 0/1 labels
        """
        
        seq_len = middle_hidden_states.shape[0]
        middle_hidden_states = middle_hidden_states.to(self.device)
        
        # Get logits
        logits = self.lm_head(middle_hidden_states)
        
        features_list = []
        labels_list = []
        
        for i in range(seq_len):
            # Extract features
            features = self.extract_features(
                logits, i,
                network_condition,
                self.current_acceptance_rate,
                draft_tokens[i] if draft_tokens else None
            )
            features_list.append(features)
            
            # Label: 1 if accepted, 0 if rejected
            label = 1.0 if i in accepted_indices else 0.0
            labels_list.append(label)
        
        return torch.stack(features_list), torch.tensor(labels_list, device=self.device)
    
    def train_step(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pruning_weight: float = 0.1
    ) -> float:
        """
        Single training step
        
        Args:
            features: [batch_size, 8] features
            labels: [batch_size] binary labels (1=keep, 0=prune)
            pruning_weight: Weight for encouraging pruning
        """
        
        self.decision_net.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.decision_net(features).squeeze()
        
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(predictions, labels)
        
        # Add small penalty to encourage pruning (for speedup)
        # This prevents the model from being too conservative
        pruning_rate = (predictions < 0.5).float().mean()
        pruning_bonus = pruning_weight * pruning_rate
        
        # Total loss
        loss = bce_loss - pruning_bonus
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.decision_net.eval()
        
        return loss.item()
    
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
            return AdaptiveNeuralPruner(hidden_size, vocab_size, config, tie_weights)
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
        """
        Main interface for pruning speculation tree
        
        Args:
            middle_hidden_states: Hidden states from middle layer
            draft_logits: Logits from draft model
            draft_tokens: Draft token IDs
            tree_structure: Tree structure information
            
        Returns:
            Pruning results with kept branches and statistics
        """
        
        # Build branch contexts from tree structure
        
        # Get network condition if using adaptive method
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
        """Save pruner state for later restoration"""
        state = {
            'config': self.config,
            'total_tokens': self.total_tokens,
            'pruned_tokens': self.pruned_tokens,
            'speedup_factor': self.speedup_factor,
            'pruner_metrics': self.pruner.get_metrics()
        }
        
        if isinstance(self.pruner, AdaptiveNeuralPruner):
            state['pruner_weights'] = self.pruner.pruning_net.state_dict()
        
        torch.save(state, path)
        logger.info(f"Saved pruner state to {path}")
    
    def load_state(self, path: str):
        """Load pruner state"""
        state = torch.load(path)
        
        self.config = state['config']
        self.total_tokens = state['total_tokens']
        self.pruned_tokens = state['pruned_tokens']
        self.speedup_factor = state['speedup_factor']
        
        # Recreate pruner with loaded config
        self.pruner = BloombeePrunerFactory.create_pruner(
            self.config.method,
            self.hidden_size,
            self.vocab_size,
            self.config
        )
        
        if isinstance(self.pruner, AdaptiveNeuralPruner) and 'pruner_weights' in state:
            self.pruner.pruning_net.load_state_dict(state['pruner_weights'])
        
        logger.info(f"Loaded pruner state from {path}")