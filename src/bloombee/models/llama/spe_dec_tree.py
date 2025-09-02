import uuid
import math
from typing import List, Optional, Dict, Tuple, Any
import torch

class TreeNode:
    """推测树的基本节点"""

    def __init__(self, token_id: int, probability: float = 1.0, depth: int = 0):
        self.token_id = token_id
        self.probability = probability
        self.depth = depth
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
        self.node_id = str(uuid.uuid4())
        self.position_in_sequence = -1

    def add_child(self, token_id: int, probability: float) -> 'TreeNode':
        """添加子节点"""
        child = TreeNode(token_id, probability, self.depth + 1)
        child.parent = self
        self.children.append(child)
        return child

    def add_children(self, candidates: List[Tuple[int, float]]) -> List['TreeNode']:
        """批量添加子节点"""
        children = []
        for token_id, prob in candidates:
            child = self.add_child(token_id, prob)
            children.append(child)
        return children

    def get_path_from_root(self) -> List[int]:
        """获取从根节点到当前节点的token路径"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.token_id)
            current = current.parent
        return list(reversed(path))

    def get_path_nodes_from_root(self) -> List['TreeNode']:
        """获取从根节点到当前节点的节点路径"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return len(self.children) == 0

    def get_all_leaf_paths(self) -> List[List[int]]:
        """获取所有从根到叶子的路径"""
        if self.is_leaf():
            return [self.get_path_from_root()]

        all_paths = []
        for child in self.children:
            child_paths = child.get_all_leaf_paths()
            all_paths.extend(child_paths)
        return all_paths

    def get_all_leaf_node_paths(self) -> List[List['TreeNode']]:
        """获取所有从根到叶子的节点路径"""
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
    """单个请求的推测树"""

    def __init__(self, root_token: int, request_id: str):
        self.root = TreeNode(root_token, 1.0, 0)
        self.request_id = request_id
        self.max_depth = 0
        self.total_nodes = 1

    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """获取指定深度的所有节点"""
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
        """为指定的父节点添加一层候选"""
        if len(parent_nodes) != len(candidates_per_node):
            raise ValueError("父节点数量与候选数量不匹配")

        new_nodes = []
        for parent, candidates in zip(parent_nodes, candidates_per_node):
            children = parent.add_children(candidates)
            new_nodes.extend(children)

        if new_nodes:
            self.max_depth = max(self.max_depth, max(node.depth for node in new_nodes))
            self.total_nodes += len(new_nodes)

        return new_nodes

    def get_all_paths(self) -> List[List[int]]:
        """获取所有可能的路径"""
        return self.root.get_all_leaf_paths()


def linearize_tree_with_positions(tree: SpeculativeTree) -> Tuple[List[TreeNode], List[int]]:
    """
    DFS线性化: 记录父位置
    """
    linearized_nodes = []
    parent_indices = []
    position_map = {}

    def dfs_with_positions(node):
        if node.parent is not None:  # 跳过root
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
    """
    使用scatter_和bit-jump优化祖先矩阵构建
    """
    n = len(parent_indices)
    if n == 0:
        return torch.empty(0, 0, dtype=torch.bool, device=device)

    # 使用scatter_一次性构建直接父子关系
    A = torch.zeros(n, n, dtype=torch.bool, device=device)

    rows = torch.arange(n, device=device)
    cols = torch.as_tensor(parent_indices, device=device)
    mask = cols >= 0  # 有效的父节点

    if mask.any():
        A[rows[mask], cols[mask]] = True

    # 使用bit-jump优化传递闭包
    ancestor_matrix = A.clone()

    # 转换为float进行矩阵运算，然后转回bool
    for _ in range(n):  # 最多n次迭代就能收敛
        # 将bool转换为float进行矩阵乘法
        A_float = A.float()
        ancestor_float = ancestor_matrix.float()

        # 计算可达关系
        power_A = torch.matmul(ancestor_float, A_float)
        new_reachable = ancestor_matrix | (power_A > 0)  # 转回bool

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
    """
    构建增量推理的attention mask:
    [tree_len, past_len + tree_len] = [left_ones, tree_ancestor_matrix]
    """
    if tree_len == 0:
        return torch.empty(0, past_len, dtype=torch.bool, device=device)

    # 左侧：树节点可以看到所有past tokens
    # left_mask = torch.ones(tree_len, past_len, dtype=torch.bool, device=device)

    # 右侧：树内部的祖先关系
    if len(parent_indices) > 0:
        ancestor_matrix = build_ancestor_matrix_optimized(parent_indices, device)
        # 树节点可以看到祖先节点 + 自己
        tree_mask = ancestor_matrix | torch.eye(tree_len, dtype=torch.bool, device=device)
    else:
        tree_mask = torch.eye(tree_len, dtype=torch.bool, device=device)

    # 拼接: [tree_len, past_len + tree_len]
    # full_mask = torch.cat([left_mask, tree_mask], dim=1)

    return tree_mask.unsqueeze(0)  # [1, tree_len, past_len + tree_len]


def prepare_incremental_tree_batch(
    trees: List[SpeculativeTree], 
    input_ids: torch.LongTensor,
    device: torch.device,
    pad_token_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[TreeNode]]]]:
    """
    为增量推理准备树tokens和attention mask
    """
    batch_size = len(trees)

    if not trees or all(tree.total_nodes <= 1 for tree in trees):
        return torch.empty(batch_size, 0, dtype=torch.long, device=device), None, [[] for _ in trees]

    max_tree_size = max(tree.total_nodes - 1 for tree in trees if tree.total_nodes > 1)
    past_len = input_ids.shape[1]

    # 收集线性化结果
    batch_tree_tokens = []
    batch_attention_masks = []
    batch_node_paths = []

    for tree in trees:
        linearized_nodes, parent_indices = linearize_tree_with_positions(tree)

        # 树tokens
        tree_token_ids = [node.token_id for node in linearized_nodes]
        padded_tokens = tree_token_ids + [pad_token_id] * (max_tree_size - len(tree_token_ids))
        batch_tree_tokens.append(padded_tokens)

        # attention mask
        tree_len = len(tree_token_ids)
        if tree_len > 0:
            mask = build_incremental_tree_attention_mask(
                past_len, tree_len, parent_indices, device
            )
            # 对padding部分补充mask
            if tree_len < max_tree_size:
                pad_len = max_tree_size - tree_len
                # padding行：只能看到past，不能看到任何树节点
                pad_mask = torch.cat([
                    torch.ones(pad_len, past_len, dtype=torch.bool, device=device),
                    torch.zeros(pad_len, max_tree_size, dtype=torch.bool, device=device)
                ], dim=1).unsqueeze(0).expand(1, pad_len, past_len + max_tree_size)
                mask = torch.cat([mask, pad_mask], dim=1)
        else:
            # 空树的情况
            mask = torch.ones(1, max_tree_size, past_len + max_tree_size, dtype=torch.bool, device=device)

        batch_attention_masks.append(mask)
        batch_node_paths.append(tree.root.get_all_leaf_node_paths())

    # 转换为tensor
    tree_tokens = torch.tensor(batch_tree_tokens, device=device)

    # 合并attention masks
    if batch_attention_masks:
        attention_mask = torch.cat(batch_attention_masks, dim=0)  # [batch_size, max_tree_size, past_len + max_tree_size]
    else:
        attention_mask = None

    return tree_tokens, attention_mask, batch_node_paths


# 兼容原版接口
def prepare_tree_attention_batch(
    trees: List[SpeculativeTree], 
    prefix_tokens: torch.Tensor,
    device: torch.device,
    pad_token_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[TreeNode]]]]:
    """
    兼容原版接口：批量处理多个树的线性化和attention mask构建
    """
    # 如果是增量模式，直接转发
    tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
        trees, prefix_tokens, device, pad_token_id
    )

    # 原版接口需要返回full_sequence
    if tree_tokens.shape[1] > 0:
        full_sequence = torch.cat([prefix_tokens, tree_tokens], dim=-1)
    else:
        full_sequence = prefix_tokens

    return full_sequence, attention_mask, batch_node_paths