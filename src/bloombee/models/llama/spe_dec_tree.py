import uuid
import math
from typing import List, Optional, Dict, Tuple, Any
import torch


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
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[TreeNode]]]]:
    """
    准备增量 tree batch，支持不同序列长度
    
    Args:
        trees: speculative trees 列表
        input_ids: [batch_size, max_seq_len] 输入 token ids（可能包含 padding）
        device: 设备
        pad_token_id: padding token id
        seq_lengths: [batch_size] 每个序列的真实长度，如果为 None 则假设所有序列长度相同
    
    Returns:
        tree_tokens: [batch_size, max_tree_size]
        attention_mask: [batch_size, max_tree_size, past_len + max_tree_size]
        batch_node_paths: 每个 batch 的节点路径列表
    """
    batch_size = len(trees)

    if not trees or all(tree.total_nodes <= 1 for tree in trees):
        return torch.empty(batch_size, 0, dtype=torch.long, device=device), None, [[] for _ in trees]

    max_tree_size = max(tree.total_nodes - 1 for tree in trees if tree.total_nodes > 1)
    past_len = input_ids.shape[1]

    batch_tree_tokens = []
    batch_attention_masks = []
    batch_node_paths = []

    for tree in trees:
        linearized_nodes, parent_indices = linearize_tree_with_positions(tree)

        tree_token_ids = [node.token_id for node in linearized_nodes]
        padded_tokens = tree_token_ids + [pad_token_id] * (max_tree_size - len(tree_token_ids))
        batch_tree_tokens.append(padded_tokens)

        tree_len = len(tree_token_ids)
        if tree_len > 0:
            mask = build_incremental_tree_attention_mask(
                past_len, tree_len, parent_indices, device
            )
            if tree_len < max_tree_size:
                pad_len = max_tree_size - tree_len
                pad_mask = torch.cat([
                    torch.ones(pad_len, past_len, dtype=torch.bool, device=device),
                    torch.zeros(pad_len, max_tree_size, dtype=torch.bool, device=device)
                ], dim=1).unsqueeze(0).expand(1, pad_len, past_len + max_tree_size)
                mask = torch.cat([mask, pad_mask], dim=1)
        else:
            mask = torch.ones(1, max_tree_size, past_len + max_tree_size, dtype=torch.bool, device=device)

        batch_attention_masks.append(mask)
        batch_node_paths.append(tree.root.get_all_leaf_node_paths())

    tree_tokens = torch.tensor(batch_tree_tokens, device=device)

    if batch_attention_masks:
        attention_mask = torch.cat(batch_attention_masks, dim=0)
    else:
        attention_mask = None

    return tree_tokens, attention_mask, batch_node_paths

def prepare_tree_attention_batch(
    trees: List[SpeculativeTree], 
    prefix_tokens: torch.Tensor,
    device: torch.device,
    pad_token_id: int = 0,
    seq_lengths: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[TreeNode]]]]:
    tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
        trees, prefix_tokens, device, pad_token_id, seq_lengths
    )
    if tree_tokens.shape[1] > 0:
        full_sequence = torch.cat([prefix_tokens, tree_tokens], dim=-1)
    else:
        full_sequence = prefix_tokens

    return full_sequence, attention_mask, batch_node_paths