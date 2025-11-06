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

class AdaptiveNeuralPruner(PrunerInterface):
    """
    Adaptive neural network based pruner
    Uses a trainable network that considers multiple factors
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
        
        # Pruning network
        self.pruning_net = self._build_pruning_network()
        
        # Optimizer for online learning
        self.optimizer = torch.optim.AdamW(
            self.pruning_net.parameters(),
            lr=config.neural_learning_rate
        )
        
        # Statistics tracking
        self.acceptance_history = deque(maxlen=100)
        self.pruning_decisions = deque(maxlen=100)
        self.latency_history = deque(maxlen=50)
        self.last_acceptances = deque(maxlen=32)
        
    def _build_pruning_network(self) -> nn.Module:
        """Build the neural network for pruning decisions"""
        input_size = 15  # Features: 6 prob + 4 network + 4 context + 1 last accept
        hidden = self.config.neural_hidden_size
        dropout = self.config.neural_dropout
        
        return nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid()
        )
    
    def extract_features(
        self,
        middle_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        context: BranchContext,
        network_condition: NetworkCondition,
        last_acceptance: bool
    ) -> torch.Tensor:
        """Extract features for the neural network"""
        
        middle_probs = F.softmax(middle_logits, dim=-1)
        draft_probs = F.softmax(draft_logits, dim=-1)
        
        # Probability features
        draft_token_prob = middle_probs[context.token_id].item()
        max_prob = torch.max(middle_probs).item()
        entropy = -torch.sum(middle_probs * torch.log(middle_probs + 1e-10)).item()
        kl_div = F.kl_div(
            middle_probs.log(),
            draft_probs,
            reduction='sum'
        ).item()
        top5_mass = torch.topk(middle_probs, k=5).values.sum().item()
        sorted_probs = torch.argsort(middle_probs, descending=True)
        draft_rank = (sorted_probs == context.token_id).nonzero(as_tuple=True)[0].item()
        draft_rank_normalized = draft_rank / len(middle_probs)
        
        # All features
        features = torch.tensor([
            # Probability features (6)
            draft_token_prob,
            max_prob,
            entropy,
            kl_div,
            top5_mass,
            draft_rank_normalized,
            # Network features (4)
            network_condition.avg_acceptance_rate,
            network_condition.recent_latency,
            network_condition.pruning_accuracy,
            min(network_condition.tokens_generated / 1000.0, 1.0),
            # Context features (4)
            context.tree_depth / 10.0,
            context.num_siblings / 20.0,
            context.parent_acceptance_prob,
            context.branch_index / max(context.num_siblings, 1),
            # Last acceptance (1)
            float(last_acceptance)
        ])
        
        return features
    
    def prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
        # if network_condition is None:
        #     network_condition = self.get_network_condition()
        
        # # Get middle layer logits
        # middle_logits = self.lm_head(middle_hidden_states)
        
        # keep_probs = []
        # for i, context in enumerate(branch_contexts):
        #     # Get last acceptance for this branch
        #     last_accept = self.last_acceptances[i] if i < len(self.last_acceptances) else True
            
        #     # Extract features
        #     features = self.extract_features(
        #         middle_logits[i] if middle_logits.dim() > 1 else middle_logits,
        #         draft_logits[i] if draft_logits.dim() > 1 else draft_logits,
        #         context,
        #         network_condition,
        #         last_accept
        #     )
            
        #     # Predict keep probability
        #     with torch.no_grad():
        #         keep_prob = self.pruning_net(features.unsqueeze(0)).squeeze().item()
        #     keep_probs.append(keep_prob)
        
        # keep_probs_tensor = torch.tensor(keep_probs)
        
        # # Apply threshold
        # keep_mask = keep_probs_tensor > self.config.neural_threshold
        
        # # Ensure constraints
        # if keep_mask.sum() < self.config.min_keep_branches:
        #     _, top_indices = torch.topk(keep_probs_tensor, self.config.min_keep_branches)
        #     keep_mask = torch.zeros_like(keep_probs_tensor, dtype=torch.bool)
        #     keep_mask[top_indices] = True
        
        # if keep_mask.sum() > self.config.max_branches:
        #     scores_masked = keep_probs_tensor * keep_mask.float()
        #     _, top_indices = torch.topk(scores_masked, self.config.max_branches)
        #     keep_mask = torch.zeros_like(keep_probs_tensor, dtype=torch.bool)
        #     keep_mask[top_indices] = True
        
        # keep_indices = torch.where(keep_mask)[0].tolist()
        # prune_indices = torch.where(~keep_mask)[0].tolist()
        
        # return {
        #     'keep_indices': keep_indices,
        #     'prune_indices': prune_indices,
        #     'keep_probs': keep_probs,
        #     'metadata': {
        #         'middle_logits': middle_logits,
        #         'network_condition': network_condition,
        #         'avg_keep_prob': sum(keep_probs) / len(keep_probs)
        #     }
        # }
    
    def get_network_condition(self) -> NetworkCondition:
        """Get current network condition"""
        return NetworkCondition(
            avg_acceptance_rate=sum(self.acceptance_history) / max(len(self.acceptance_history), 1),
            recent_latency=sum(self.latency_history) / max(len(self.latency_history), 1),
            tokens_generated=len(self.acceptance_history),
            pruning_accuracy=sum(self.pruning_decisions) / max(len(self.pruning_decisions), 1)
        )
    
    def update_statistics(self, feedback: Dict[str, Any]):
        """Update statistics and optionally train the network"""
        if 'acceptance' in feedback:
            self.acceptance_history.append(float(feedback['acceptance']))
        if 'latency' in feedback:
            self.latency_history.append(feedback['latency'])
        if 'pruning_correct' in feedback:
            self.pruning_decisions.append(float(feedback['pruning_correct']))
        if 'branch_acceptances' in feedback:
            self.last_acceptances.extend(feedback['branch_acceptances'])
        
        # Online training if enough data
        if 'training_data' in feedback and len(self.acceptance_history) > 10:
            self._train_step(feedback['training_data'])
    
    def _train_step(self, training_data: Dict[str, Any]):
        """Perform a single training step"""
        self.optimizer.zero_grad()
        
        features = training_data['features']
        targets = training_data['targets']
        
        predictions = self.pruning_net(features)
        loss = F.binary_cross_entropy(predictions, targets)
        
        # Add speedup bonus
        num_pruned = (predictions < self.config.neural_threshold).sum()
        speedup_bonus = (num_pruned / len(predictions)) * self.config.neural_speedup_weight
        total_loss = loss - speedup_bonus
        
        total_loss.backward()
        self.optimizer.step()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            'avg_acceptance_rate': sum(self.acceptance_history) / max(len(self.acceptance_history), 1),
            'pruning_accuracy': sum(self.pruning_decisions) / max(len(self.pruning_decisions), 1),
            'avg_latency': sum(self.latency_history) / max(len(self.latency_history), 1),
            'history_size': len(self.acceptance_history)
        }


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


# Example usage and integration
if __name__ == "__main__":
    # Configuration
    hidden_size = 4096
    vocab_size = 32000
    
    # Create configuration
    config = PruningConfig(
        method=PruningMethod.SIMPLE_PROBABILITY,
        neural_threshold=0.5,
        simple_threshold=0.3
    )
    
    # Create manager
    manager = BloombeePrunerManager(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        config=config
    )
    
    # Example: Prune a speculation tree
    batch_size = 1
    num_branches = 10
    
    middle_hidden = torch.randn(batch_size, num_branches, hidden_size)
    draft_logits = torch.randn(batch_size, num_branches, vocab_size)
    draft_tokens = [i for i in range(num_branches)]
    
    tree_structure = {
        'depths': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        'num_siblings': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        'parent_probs': [1.0, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7],
        'positions': list(range(num_branches))
    }
    
    # Perform pruning
    results = manager.prune_speculation_tree(
        middle_hidden[0],  # Remove batch dimension for this example
        draft_logits[0],
        draft_tokens,
        tree_structure
    )
    
    print(f"Kept branches: {results['keep_indices']}")
    print(f"Pruned branches: {results['prune_indices']}")
    print(f"Keep probabilities: {results['keep_probs']}")
    print(f"Manager metrics: {results['manager_metrics']}")
    
    # Switch to simple method
    manager.switch_method(PruningMethod.SIMPLE_PROBABILITY)
    
    # Perform pruning with simple method
    results_simple = manager.prune_speculation_tree(
        middle_hidden[0],
        draft_logits[0],
        draft_tokens,
        tree_structure
    )
    
    print(f"\nSimple method results:")
    print(f"Kept branches: {results_simple['keep_indices']}")
    
    # Get comprehensive metrics
    metrics = manager.get_comprehensive_metrics()
    print(f"\nComprehensive metrics: {metrics}")