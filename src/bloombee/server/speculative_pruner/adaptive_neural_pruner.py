import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import math

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
    
    def forward(self, prob_features):
        quality_score = self.quality_path(prob_features).squeeze(-1)  # [batch]
        decision_score = quality_score
        decision_prob = torch.sigmoid(decision_score)
        
        return decision_prob, quality_score

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
        self.lm_head = SimpleLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to("cuda")
        checkpoint_path = "checkpoints/lmhead/lm_head_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        if 'model_state_dict' in checkpoint:
            self.lm_head.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果你只存了 state_dict 本身
            self.lm_head.load_state_dict(checkpoint)
            
        self.lm_head.requires_grad_(False)
        self.lm_head.to(dtype=torch.float16)
        
        self.final_lm_head = SimpleLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to("cuda")
        self.final_lm_head.load_weight("/tmp/data/llama_weights/llama-7b-np")
        self.final_lm_head.requires_grad_(False)
        self.final_lm_head.to(dtype=torch.float16)
        
        # Dual-path decision network
        self.decision_net = DualPathPruner(hidden_size=neural_hidden).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.decision_net.parameters(), 
            lr=1e-4
        )
        
        # Historical acceptance rate tracking
        self.acceptance_history = deque(maxlen=100)
        self.current_acceptance_rate = 1
        
        # Training mode flag
        self.training = False
        
        # self.collector = TrainingDataCollector(save_dir="./pruning_data")
        self.ite = 0
        self.g_ite = 0
        
        self.temp_ite_count = 0
        self.atc = 0 # accept_tokens_count
        self.after_pruing_atc = 0
        self.keep_count = 0
        
        self.simple_keep_count = 0
        self.after_simple_pruning_atc = 0
    
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
        max_prob = torch.max(probs)
        # logger.info(f"max_prob: {max_prob}")
        entropy = -torch.sum(torch.where(
            probs > 1e-10,
            probs * torch.log(probs),
            torch.zeros_like(probs)
        ))
        # logger.info(f"entropy: {entropy}")
        log_vocab_size = torch.log(torch.tensor(float(self.vocab_size)))
        # logger.info(f"log_vocab_size: {log_vocab_size}")
        if log_vocab_size > 0:
            normalized_entropy = (entropy / log_vocab_size)
        else:
            normalized_entropy = 0.0  # vocab_size <= 1 时
        # logger.info(f"normalized_entropy: {normalized_entropy}")
        
        # Token概率
        if draft_token is not None:
            token_prob = probs[draft_token]
        else:
            token_prob = torch.topk(probs, k=min(5, self.vocab_size)).values.sum()
            
        eps = 1e-10
        logp_draft = token_prob + eps
        log_ratio  = logp_draft
        log_ratio = max(log_ratio, -10.0) / 10.0
        log_ratio = -log_ratio
        
        # 组合特征
        prob_features = torch.stack([
            max_prob, 
            normalized_entropy, 
            log_ratio,
        ])
        
        network_features = torch.tensor(
            network_condition.to_features(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        
        return prob_features, network_features
    
    def _get_parent_postion(self, i, mask, prefix):
        for j in range(i-1, -1, -1):
            if mask[0, i, j + prefix] == True:
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
        # norm_middle_hidden_states = 
        logits = self.lm_head(middle_hidden_states)
        
        # Initialize masks
        keep_mask = torch.ones(seq_len, dtype=torch.bool)
        discarded = torch.zeros(seq_len, dtype=torch.bool)
        decision_probs = torch.zeros(seq_len)
        quality_scores = torch.zeros(seq_len)
        threshold_adjusts = torch.zeros(seq_len)
        
        # logger.info(f"prune_branches, seq_len: {seq_len}")
        # logger.info(f"prune_branches, discarded: {discarded}")
        
        # Process each position
        prob_features_list, network_features_list, _ = self.collect_training_data(
                middle_hidden_states, tree_attention_mask, None, 
                network_condition,
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
            
            # Extract features
            # parent_postion = self._get_parent_postion(i, tree_attention_mask, prefix_len)
            prob_features = prob_features_list[i-1]
            network_features = network_features_list[i-1]
            
            parent_postion = self._get_parent_postion(i, tree_attention_mask, prefix_len)
            # logger.info(f"xiongxu i: {i}, parent_postion: {parent_postion}")
            logits_at_pos = logits[0, parent_postion]
            # logger.info(f"xiongxu i: {i}, logits_at_pos: {logits_at_pos}")
            probs = F.softmax(logits_at_pos, dim=-1)
            # logger.info(f"xiongxu i: {i}, probs: {probs}")
            token_prob = probs[draft_tokens[i]].item()  # ✅ 转为 Python float
            # logger.info(f"xiongxu draft_token i {i},  {draft_tokens[i]} , prob: {token_prob}")
            # logger.info(f"prune_branches, prob_features: {prob_features}")
            # logger.info(f"prune_branches, network_features: {network_features}")
            
            # Get decision from dual-path network
            with torch.no_grad():
                prob, quality, threshold = self.decision_net(
                    prob_features.unsqueeze(0),
                    network_features.unsqueeze(0)
                )
                
                decision_probs[i] = prob.item()
                quality_scores[i] = quality.item()
                threshold_adjusts[i] = 0
                
                # Decision: >0.5 means keep
                keep = prob.item() > self.config.neural_threshold
                
                # logger.info(f"prune_branches, prob: {prob}, quality: {quality}, prob_features: {prob_features}, i: {i}")
            
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
        
        # Get final indices
        keep_indices = torch.where(keep_mask)[0].tolist()
        prune_indices = torch.where(~keep_mask)[0].tolist()
        
        return {
            'keep_indices': keep_indices,
            'prune_indices': prune_indices,
            'decision_probs': decision_probs.cpu().tolist(),
            'quality_scores': quality_scores.cpu().tolist(),
            'threshold_adjusts': 0,
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
        
        seq_len = middle_hidden_states.shape[1]
        
        prefix_len = tree_attention_mask.shape[2] - seq_len
        
        # ✅ 在 no_grad 中提取所有数值
        with torch.no_grad():
            middle_hidden_states = middle_hidden_states.to(self.device)
            logits = self.lm_head(middle_hidden_states)
            
            prob_features_list = []
            labels_list = []
            
            acceptance_rate_value = self.current_acceptance_rate  # Python float
            network_features_values = network_condition.to_features()  # Python list/array
            
            for i in range(1, seq_len):
                parent_postion = self._get_parent_postion(i, tree_attention_mask, prefix_len)
                
                logits_at_pos = logits[0, parent_postion]
                probs = F.softmax(logits_at_pos, dim=-1)
                
                
                max_prob = torch.max(probs).item()  # ✅ 转为 Python float
                
                entropy = -torch.sum(torch.where(
                    probs > 1e-10,
                    probs * torch.log(probs),
                    torch.zeros_like(probs)
                )).item()  # ✅ 转为 Python float
                
                log_vocab_size = math.log(float(self.vocab_size))
                if log_vocab_size > 0:
                    normalized_entropy = entropy / log_vocab_size
                else:
                    normalized_entropy = 0.0
                
                if draft_tokens is not None:
                    token_prob = probs[draft_tokens[i]].item()  # ✅ 转为 Python float
                else:
                    token_prob = torch.topk(probs, k=min(5, self.vocab_size)).values.sum().item()
                    
                eps = 1e-10
                logp_draft = math.log(token_prob + eps)
                log_ratio  = logp_draft
                log_ratio = max(log_ratio, -10.0) / 10.0
                log_ratio = -log_ratio
                
                # ✅ 存储 Python 值
                prob_features_list.append([
                    max_prob,
                    normalized_entropy,
                    log_ratio,
                ])

                top200_results = torch.topk(probs, k=100, dim=-1)
                top200_ids = top200_results.indices.tolist()

                draft_token_id = draft_tokens[i].item()

                if draft_token_id in top200_ids:
                    # 获取该 token 的具体排名
                    rank = top200_ids.index(draft_token_id) + 1 # rank 从 1 开始
                    
                    label = 1.0 - rank / 100.0
                else:
                    label = 0.0

                # 最后再乘上你的深度衰减系数（如果你还要保留深度控制的话）
                labels_list.append(label)
        
        # ✅ 在 no_grad 外面创建新张量（带梯度）
        prob_features = torch.tensor(
            prob_features_list, 
            dtype=torch.float32, 
            device=self.device,
        )
        
        network_features = torch.tensor(
            network_features_values,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).repeat(seq_len, 1)
        
        labels = torch.tensor(
            labels_list, 
            dtype=torch.float32, 
            device=self.device
        )
        
        return prob_features, network_features, labels
        
    def _get_current_accepted_tokens_indices(
            self, 
            final_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            draft_tokens: torch.Tensor,
        ):
        """
        final_hidden_states: [B, seq_len, hidden_dim]
        attention_mask: [B, seq_len, seq_len + prefix_len]
        draft_tokens: [B, seq_len]
        """
        B, seq_len, _ = final_hidden_states.shape
        prefix_len = attention_mask.shape[2] - seq_len

        # logits = [1, seq_len, vocab]
        logits = self.final_lm_head(final_hidden_states)
        # logger.info(f"_get_current_accepted_tokens_indices, logits: {logits}")
        probs = torch.softmax(logits, dim=-1)
        # logger.info(f"_get_current_accepted_tokens_indices, probs: {probs}")

        # ==================================
        # Step 1 — 根据 attention mask 还原每条 root→leaf path
        # ==================================
        is_leaf = torch.ones(seq_len, dtype=torch.bool)
        leaf_paths = []
        
        # logger.info(f"_get_current_accepted_tokens_indices, seq_len: {seq_len}, prefix_len: {prefix_len}")

        for i in range(seq_len - 1, -1, -1):
            if is_leaf[i]:
                path = [i]

                # 回溯依赖链
                for j in range(i - 1, -1, -1):
                    # attention_mask[0, child, parent+prefix] == 1
                    if attention_mask[0, i, j + prefix_len] == 1:
                        is_leaf[j] = False
                        path.append(j)

                path.reverse()
                leaf_paths.append((i, path))

        # ==================================
        # Step 2 — 对每个 path 做验证（root 默认成功）
        # ==================================
        best_path = None
        best_validated = -1
        
        # logger.info(f"_get_current_accepted_tokens_indices, leaf_paths {leaf_paths}")

        for leaf_idx, path in leaf_paths:
            validated = 1   # root always validated
            # logger.info(f"current path : {path}")
            for i in range(1, len(path)):
                idx = path[i]
                token_id = draft_tokens[idx].item()
                pred_id = probs[0, path[i - 1]].argmax().item()
                # logger.info(f"current i: {i}, token_id: {token_id}, pred_id: {pred_id}")

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
    
    def simple_prune_branches(
        self,
        middle_hidden_states: torch.Tensor,
        draft_tokens: List[int],
        tree_attention_mask: torch.Tensor,
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
        
        # Ensure minimum branches are kept
        kept_count = keep_mask.sum().item()
        
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
    
    def train_step(
        self,
        middle_hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
        alpha: float = 1.0,               # BCE loss weight
        beta: float = 0.0,                 # Pruning alignment weight
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

        
        
        accepted_indices, best_validated = self._get_current_accepted_tokens_indices(final_hidden_states, attention_mask, draft_tokens)
        logger.info(f"train_step, accepted_indices: {accepted_indices}")
        prob_features, network_features, labels = self.collect_training_data(
            middle_hidden_states,                
            attention_mask, 
            accepted_indices, 
            NetworkCondition.mock(), 
            draft_tokens)
        
        # self.collector.add_batch(prob_features, labels, metadata={'iteration': self.ite})
        self.ite = self.ite + 1
        
        self.decision_net.train()
        self.optimizer.zero_grad()
        
        tree_size = draft_tokens.shape[0]
        
        with torch.enable_grad():
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
            
        logger.info(f"predictions: {predictions}")
        logger.info(f"labels: {labels}")
        
        bce_loss = F.binary_cross_entropy(
            predictions, 
            labels, 
            weight=sample_weights
        )
        
        logger.info(f"train_step, bce_loss: {bce_loss}")
        
        # === Loss 2: Network-Aware Pruning Alignment ===
        # 当前预测的pruning rate
        current_pruning_rate = torch.sigmoid((0.5 - predictions) * 10).mean()
        
        # 从网络特征计算目标pruning rate
        # 使用第一个token的网络特征（整棵树共享）
        bandwidth = network_features[0, 0].item()
        network_severity = 1.0 - min(bandwidth / 100.0, 1.0)
        target_pruning_rate = 0.2 + 0.6 * network_severity  # 20%-80%
        
        # Alignment loss
        pruning_alignment = (current_pruning_rate - target_pruning_rate) ** 2
        
        # === Total Loss ===
        total_loss = alpha * bce_loss + beta * pruning_alignment
        
        logger.info(f"train_step, total_loss: {total_loss}")
        
        total_loss.backward()
        
        self.optimizer.step()
        
        self.decision_net.eval()
        
        # update acceptance rate
        self.temp_ite_count += 1
        self.atc += len(accepted_indices) - 1
        results = self.prune_branches(
            middle_hidden_states,
            draft_tokens,
            attention_mask,
            None,
        )
        
        simple_results = self.simple_prune_branches(
            middle_hidden_states,
            draft_tokens,
            attention_mask,
        )
        
        keep_indices = results['keep_indices']
        count = 0
        for indice in accepted_indices:
            if indice in keep_indices:
                count += 1
        self.after_pruing_atc += count - 1
        self.keep_count += len(keep_indices) - 1
        logger.info(f"pruning rate: {self.keep_count / (self.temp_ite_count * 30)}")
        logger.info(f"before pruning, accept rate: {self.atc / (self.temp_ite_count * 4)}")
        logger.info(f"after pruning, accept rate: {self.after_pruing_atc / (self.temp_ite_count * 4)}")
        
        simple_keep_indices = simple_results['keep_indices']
        self.simple_keep_count += len(simple_keep_indices) - 1
        count = 0
        for indice in accepted_indices:
            if indice in simple_keep_indices:
                count += 1
        self.after_simple_pruning_atc += count - 1
        logger.info(f"simple pruning rate: {self.simple_keep_count / (self.temp_ite_count * 30)}")
        logger.info(f"before simple pruning, accept rate: {self.atc / (self.temp_ite_count * 4)}")
        logger.info(f"after simple pruning, accept rate: {self.after_simple_pruning_atc / (self.temp_ite_count * 4)}")
        
        return {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'pruning_alignment': pruning_alignment.item(),
            'current_pruning_rate': current_pruning_rate.item(),
            'target_pruning_rate': target_pruning_rate,
            'network_severity': network_severity,
            'avg_quality_score': quality_scores.mean().item(),
            'avg_threshold': 0,
            'pos_count': pos_count.item(),
            'neg_count': neg_count.item()
        }
    
    def update_acceptance_rate(self, rate: float):
        self.current_acceptance_rate = rate
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'decision_net': self.decision_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'lm_head': self.lm_head.state_dict(),
            'acceptance_history': list(self.acceptance_history),
            'current_acceptance_rate': self.current_acceptance_rate,
            'ite': self.ite,
            'g_ite': self.g_ite,
        }, path)
        # self.collector.save_to_csv(f"data_iter_{self.g_ite+1}.csv")
        self.g_ite = self.g_ite + 1
        # self.collector.clear()
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.decision_net.load_state_dict(checkpoint['decision_net'])
        if 'optimizer' in checkpoint:  # ✅ 添加
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            
        # self.lm_head.load_state_dict(checkpoint['lm_head'])
        self.acceptance_history = deque(checkpoint['acceptance_history'], maxlen=100)
        self.current_acceptance_rate = checkpoint['current_acceptance_rate']
        
        if 'ite' in checkpoint:  # ✅ 添加
            self.ite = checkpoint['ite']
        if 'g_ite' in checkpoint:
            self.g_ite = checkpoint['g_ite']
    
    def get_network_condition(self):
        """Override this in subclass if needed"""
        return NetworkCondition.mock()
