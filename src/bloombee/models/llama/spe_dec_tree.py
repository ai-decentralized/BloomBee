import uuid
import math
from typing import List, Optional, Dict, Tuple, Any
import torch

from hivemind.utils.logging import get_logger

logger = get_logger()


class TreeNode:
    def __init__(self, token_id: int, probability: float = 1.0, depth: int = 0):
        self.token_id = token_id
        self.probability = probability
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
        self.node_id = str(uuid.uuid4())
        self.position_in_sequence = -1

    def add_child(self, token_id: int, probability: float) -> 'TreeNode':
        child = TreeNode(token_id, probability, self.depth + 1)
        child.parent = self
        self.children.append(child)
        return child

    def add_children(self, candidates: List[Tuple[int, float]]) -> List['TreeNode']:
        children = []
        for token_id, prob in candidates:
            child = self.add_child(token_id, prob)
            children.append(child)
        return children

    def get_path_from_root(self) -> List[int]:
        path = []
        current = self
        while current.parent is not None:
            path.append(current.token_id)
            current = current.parent
        return list(reversed(path))

    def get_path_nodes_from_root(self) -> List['TreeNode']:
        path = []
        current = self
        while current.parent is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_all_leaf_paths(self) -> List[List[int]]:
        if self.is_leaf():
            return [self.get_path_from_root()]

        all_paths = []
        for child in self.children:
            child_paths = child.get_all_leaf_paths()
            all_paths.extend(child_paths)
        return all_paths

    def get_all_leaf_node_paths(self) -> List[List['TreeNode']]:
        if self.is_leaf():
            return [self.get_path_nodes_from_root()]

        all_paths = []
        for child in self.children:
            child_paths = child.get_all_leaf_node_paths()
            all_paths.extend(child_paths)
        return all_paths

    def __str__(self):
        return f"TreeNode(token={self.token_id}, prob={self.probability:.3f}, depth={self.depth})"

class SpeculativeTree:
    def __init__(self, root_token: int, request_id: str):
        self.root = TreeNode(root_token, 1.0, 0)
        self.request_id = request_id
        self.max_depth = 0
        self.total_nodes = 1

    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        if depth == 0:
            return [self.root]

        nodes = []
        def traverse(node, current_depth):
            if current_depth == depth:
                nodes.append(node)
            elif current_depth < depth:
                for child in node.children:
                    traverse(child, current_depth + 1)

        traverse(self.root, 0)
        return nodes

    def add_layer(self, parent_nodes: List[TreeNode], candidates_per_node: List[List[Tuple[int, float]]]):
        if len(parent_nodes) != len(candidates_per_node):
            raise ValueError("parent_nodes and candidates_per_node must have the same length")

        new_nodes = []
        for parent, candidates in zip(parent_nodes, candidates_per_node):
            children = parent.add_children(candidates)
            new_nodes.extend(children)

        if new_nodes:
            self.max_depth = max(self.max_depth, max(node.depth for node in new_nodes))
            self.total_nodes += len(new_nodes)

        return new_nodes

    def get_all_paths(self) -> List[List[int]]:
        return self.root.get_all_leaf_paths()


def linearize_tree_with_positions(tree: SpeculativeTree) -> Tuple[List[TreeNode], List[int]]:
    linearized_nodes = []
    parent_indices = []
    position_map = {}

    def dfs_with_positions(node):
        if node.parent is not None:
            pos = len(linearized_nodes)
            position_map[node] = pos
            node.position_in_sequence = pos
            linearized_nodes.append(node)

            parent_pos = position_map.get(node.parent, -1)
            parent_indices.append(parent_pos)

        for child in node.children:
            dfs_with_positions(child)

    dfs_with_positions(tree.root)
    return linearized_nodes, parent_indices


def build_ancestor_matrix_optimized(parent_indices: List[int], device: torch.device) -> torch.Tensor:
    n = len(parent_indices)
    if n == 0:
        return torch.empty(0, 0, dtype=torch.bool, device=device)

    A = torch.zeros(n, n, dtype=torch.bool, device=device)

    rows = torch.arange(n, device=device)
    cols = torch.as_tensor(parent_indices, device=device)
    mask = cols >= 0

    if mask.any():
        A[rows[mask], cols[mask]] = True

    ancestor_matrix = A.clone()

    for _ in range(n):
        A_float = A.float()
        ancestor_float = ancestor_matrix.float()

        power_A = torch.matmul(ancestor_float, A_float)
        new_reachable = ancestor_matrix | (power_A > 0)

        if torch.equal(new_reachable, ancestor_matrix):
            break

        ancestor_matrix = new_reachable

    return ancestor_matrix


def build_incremental_tree_attention_mask(
    past_len: int,
    tree_len: int,
    parent_indices: List[int],
    device: torch.device
) -> torch.Tensor:
    if tree_len == 0:
        return torch.empty(0, past_len, dtype=torch.bool, device=device)

    if len(parent_indices) > 0:
        ancestor_matrix = build_ancestor_matrix_optimized(parent_indices, device)
        tree_mask = ancestor_matrix | torch.eye(tree_len, dtype=torch.bool, device=device)
    else:
        tree_mask = torch.eye(tree_len, dtype=torch.bool, device=device)

    return tree_mask.unsqueeze(0)

def prepare_incremental_tree_batch(
    trees: List[SpeculativeTree], 
    input_ids: torch.LongTensor,
    device: torch.device,
    pad_token_id: int = 0,
    seq_lengths: Optional[torch.LongTensor] = None,
    is_prefill: bool = False,
    kv_cache_position_ids: Optional[torch.Tensor] = None,  # (B, max_pos_len), -1 是 padding
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[TreeNode]]]]:
    """
    准备增量 tree batch，支持不同序列长度
    """
    batch_size = len(trees)

    if not trees or all(tree.total_nodes <= 1 for tree in trees):
        return torch.empty(batch_size, 0, dtype=torch.long, device=device), None, [[] for _ in trees]

    max_tree_size = max(tree.total_nodes - 1 for tree in trees if tree.total_nodes > 1)
    
    # Generation 阶段：计算统一的 cache_len（所有 batch 中最大的有效位置 + 1）
    cache_len = 0
    if not is_prefill and kv_cache_position_ids is not None and kv_cache_position_ids.numel() > 0:
        valid_mask = kv_cache_position_ids >= 0
        if valid_mask.any():
            max_position = kv_cache_position_ids[valid_mask].max().item()
            cache_len = int(max_position) + 1

    batch_tree_tokens = []
    batch_attention_masks = []
    batch_node_paths = []

    for i, tree in enumerate(trees):
        if seq_lengths is not None:
            curr_seq_len = seq_lengths[i].item()
        else:
            curr_seq_len = input_ids.shape[1]
        
        linearized_nodes, parent_indices = linearize_tree_with_positions(tree)

        tree_token_ids = [node.token_id for node in linearized_nodes]
        padded_tokens = tree_token_ids + [pad_token_id] * (max_tree_size - len(tree_token_ids))
        batch_tree_tokens.append(padded_tokens)

        tree_len = len(tree_token_ids)  # 不包含 root
        inputs_len = tree_len + 1  # root + tree tokens
        
        if is_prefill:
            # ============ Prefill 阶段（不变） ============
            past_len = input_ids.shape[1]
            total_len = past_len + tree_len
            mask = torch.zeros(1, total_len, total_len, dtype=torch.bool, device=device)
            
            prompt_len = curr_seq_len - 1 if curr_seq_len > 0 else 0
            root_pos = prompt_len
            
            if prompt_len > 0:
                row_idx = torch.arange(prompt_len, device=device).view(-1, 1)
                col_idx = torch.arange(prompt_len, device=device).view(1, -1)
                causal_mask = row_idx >= col_idx
                mask[0, :prompt_len, :prompt_len] = causal_mask
            
            if prompt_len > 0:
                mask[0, root_pos, :prompt_len] = True
            mask[0, root_pos, root_pos] = True
            
            if tree_len > 0:
                if prompt_len > 0:
                    mask[0, past_len:past_len + tree_len, :prompt_len] = True
                mask[0, past_len:past_len + tree_len, root_pos] = True
            
            if tree_len > 0:
                tree_mask = build_tree_attention_mask_with_root(tree_len, parent_indices, device)
                mask[0, past_len:past_len + tree_len, past_len:past_len + tree_len] = tree_mask
            
            if tree_len < max_tree_size:
                total_padded_len = past_len + max_tree_size
                padded_mask = torch.zeros(1, total_padded_len, total_padded_len, dtype=torch.bool, device=device)
                padded_mask[0, :total_len, :total_len] = mask[0]
                if curr_seq_len > 0:
                    padded_mask[0, total_len:, :curr_seq_len] = True
                mask = padded_mask
        
        else:
            # ============ Generation 阶段 ============
            # 总长度 = cache + 本轮输入
            total_len = cache_len + inputs_len
            
            mask = torch.zeros(1, inputs_len, total_len, dtype=torch.bool, device=device)
            
            # 计算 cache 中的有效位置
            cache_valid_mask = _compute_single_cache_valid_mask(
                kv_cache_position_ids[i], cache_len, device
            )
            
            # 1. Root attend to cache + 自己
            mask[0, 0, :cache_len] = cache_valid_mask
            mask[0, 0, cache_len] = True  # root attend 自己
            
            # 2. Tree tokens attend to cache + root
            if tree_len > 0:
                mask[0, 1:inputs_len, :cache_len] = cache_valid_mask.unsqueeze(0).expand(tree_len, cache_len)
                mask[0, 1:inputs_len, cache_len] = True  # tree tokens attend to root
            
            # 3. Tree tokens 之间
            if tree_len > 0:
                tree_mask = build_tree_attention_mask_with_root(tree_len, parent_indices, device)
                mask[0, 1:inputs_len, cache_len + 1:total_len] = tree_mask
            
            # Padding
            max_inputs_len = max_tree_size + 1
            if inputs_len < max_inputs_len:
                pad_len = max_inputs_len - inputs_len
                total_padded_len = cache_len + max_inputs_len
                padded_mask = torch.zeros(1, max_inputs_len, total_padded_len, dtype=torch.bool, device=device)
                padded_mask[0, :inputs_len, :total_len] = mask[0]
                # Padding 行 attend to cache（避免 NaN）
                padded_mask[0, inputs_len:, :cache_len] = cache_valid_mask.unsqueeze(0).expand(pad_len, cache_len)
                mask = padded_mask

        batch_attention_masks.append(mask)
        batch_node_paths.append(tree.root.get_all_leaf_node_paths())

    tree_tokens = torch.tensor(batch_tree_tokens, device=device)

    if batch_attention_masks:
        attention_mask = torch.cat(batch_attention_masks, dim=0)
    else:
        attention_mask = None

    return tree_tokens, attention_mask, batch_node_paths


def _compute_single_cache_valid_mask(
    kv_cache_position_ids_single: torch.Tensor,  # (max_pos_len,) 单个 batch
    cache_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    计算单个 batch 的 cache 有效位置 mask
    
    Cache 布局：
    [已整理好的 cache: 0 到 root_pos-1] [上一轮的 tree (含空洞): root_pos 到 cache_len-1]
    
    Returns:
        cache_valid_mask: (cache_len,) - True 表示有效位置
    """
    kv_cache_position_ids_single = kv_cache_position_ids_single.to(device)
    
    valid_mask = kv_cache_position_ids_single >= 0
    
    cache_valid_mask = torch.zeros(cache_len, dtype=torch.bool, device=device)
    
    # 1. 找到 root_position（第一个有效值）
    first_valid_idx = valid_mask.int().argmax().item()
    root_position = kv_cache_position_ids_single[first_valid_idx].item()
    
    # [0, root_position) 一定有效（已整理好的部分）
    cache_valid_mask[:root_position] = True
    
    # 2. kv_cache_position_ids 中的有效位置（上一轮被接收的 token）
    valid_positions = kv_cache_position_ids_single[valid_mask]
    valid_positions = valid_positions.clamp(0, cache_len - 1)
    cache_valid_mask[valid_positions] = True
    
    return cache_valid_mask


def build_tree_attention_mask_with_root(
    tree_len: int,
    parent_indices: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    构建 tree tokens 之间的 attention mask（不包含 root）
    """
    mask = torch.zeros(tree_len, tree_len, dtype=torch.bool, device=device)
    
    for i in range(tree_len):
        mask[i, i] = True
        current = i
        while current >= 0:
            parent = parent_indices[current]
            if parent >= 0:
                mask[i, parent] = True
                current = parent
            else:
                break
    
    return mask