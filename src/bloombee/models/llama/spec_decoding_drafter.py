import math
import os
import torch
from bloombee.models.llama.spe_dec_tree import SpeculativeTree
from bloombee.models.llama.spec_decoding_tree_shape import (
    FrontierNode,
    budgeted_expand_plan,
)
from typing import Dict, List, Optional, Sequence, Tuple, Union
import threading
from transformers.cache_utils import DynamicCache
import time

from hivemind.utils.logging import get_logger

logger = get_logger()


# Family-specific drafter registry. A drafter's acceptance rate depends
# on how well it models the target's output distribution — a Llama-68m
# drafter for a Qwen3 target wastes most proposals because the vocab
# and training distribution don't line up. We only route by target
# model_type, which is what HF gives us from AutoConfig, and we only
# pick drafters whose tokenizer matches the target's family (same
# vocab → draft tokens are meaningful at the target).
#
# Each entry is the list of drafter HF ids (order matters: first entry
# is the default). Callers override via MultiSSMDrafter(ssm_model_name=)
# or the BLOOMBEE_DRAFTER env var.
#
# Note on llama-68m: its HF repo only ships pytorch_model.bin (no
# safetensors). transformers 5.x on torch<2.6 refuses to load .bin
# files per CVE-2025-32434, so we default to llama-160m which has
# safetensors. llama-68m is still selectable via BLOOMBEE_DRAFTER
# when the environment supports it.
_DEFAULT_DRAFTER = "JackFram/llama-160m"

_DRAFTER_REGISTRY: Dict[str, Tuple[str, ...]] = {
    # Llama / Llama-2 / Vicuna / TinyLlama — all use the Llama tokenizer.
    "llama": ("JackFram/llama-160m", "JackFram/llama-68m"),
    # Llama-3 uses a different BPE vocab (tiktoken-based), so Llama-2
    # drafters won't be aligned. Use Meta's own 1B/3B instruct drafters.
    "llama3": ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B"),
    # Qwen3 dense (0.6B, 1.7B, 4B) share the Qwen3 tokenizer with
    # Qwen3-14B/32B targets, so acceptance is well-defined.
    "qwen3": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"),
    # Qwen3-MoE targets — no small MoE sibling, fall back to dense Qwen3
    # drafters which share the tokenizer.
    "qwen3_moe": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"),
    # Bloom has genuine small siblings in its own tokenizer family.
    "bloom": ("bigscience/bloom-560m", "bigscience/bloom-1b1"),
    # Falcon — prefer Falcon3 1B/3B for speculator alignment.
    "falcon": ("tiiuae/Falcon3-1B-Instruct", "tiiuae/falcon-rw-1b"),
    # Mixtral: no small Mixtral; use Mistral dense as an approximate
    # family sibling (same tokenizer).
    "mixtral": ("mistralai/Mistral-7B-v0.3",),
}


def select_drafter_for_target(
    target_model_type: Optional[str] = None,
    target_name_or_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Pick a drafter HF id for a given target.

    Returns (drafter_id, source) where ``source`` is one of
    ``"env"`` / ``"registry:{model_type}"`` / ``"default"`` so the
    caller can log why a specific drafter was chosen.

    Env override: ``BLOOMBEE_DRAFTER`` short-circuits the registry.
    """
    env_override = os.environ.get("BLOOMBEE_DRAFTER")
    if env_override:
        return env_override, "env"

    mt = (target_model_type or "").lower()
    # Llama-3 vs Llama-2 share model_type="llama" in HF; disambiguate on
    # the name string. Paths like "meta-llama/Meta-Llama-3-*" or
    # "meta-llama/Llama-3.2-*" pick the Llama-3 tokenizer family.
    if mt == "llama" and target_name_or_path:
        lower_path = target_name_or_path.lower()
        if "llama-3" in lower_path or "llama3" in lower_path:
            return _DRAFTER_REGISTRY["llama3"][0], "registry:llama3"

    if mt in _DRAFTER_REGISTRY:
        return _DRAFTER_REGISTRY[mt][0], f"registry:{mt}"

    return _DEFAULT_DRAFTER, "default"


def _node_path_log_prob(node) -> float:
    # Walk root→node (root has prob 1.0 so log=0). Used by EAGLE-2 ranking.
    total = 0.0
    cur = node
    while cur is not None and cur.parent is not None:
        if cur.probability > 0:
            total += math.log(cur.probability)
        else:
            return float("-inf")
        cur = cur.parent
    return total

class MultiSSMDrafter:

    def __init__(self, ssm_model_name: str, num_workers: int = 2, device: str = 'cuda'):
        from transformers import AutoModelForCausalLM

        self.num_workers = num_workers
        self.device = torch.device(device)
        self.ssm_model_name = ssm_model_name

        self.ssms = []
        self.streams = []
        for _ in range(num_workers):
            ssm = AutoModelForCausalLM.from_pretrained(
                ssm_model_name,
                torch_dtype=torch.float16)
            ssm = ssm.to(self.device)
            ssm.eval()
            self.ssms.append(ssm)
            self.streams.append(torch.cuda.Stream(device=self.device))

        with torch.no_grad():
            dummy = torch.ones(1, 8, dtype=torch.long, device=self.device)
            for ssm in self.ssms:
                ssm(dummy, attention_mask=torch.ones_like(dummy))

    @classmethod
    def for_target(
        cls,
        target: Union[str, "object"],
        *,
        num_workers: int = 2,
        device: str = 'cuda',
    ) -> "MultiSSMDrafter":
        """Construct a drafter matched to ``target``'s model family.

        ``target`` can be an HF model-id string / local path, a loaded
        `PretrainedConfig`, or a loaded model. We resolve ``model_type``
        via AutoConfig if the input is a string. Falls back to
        ``JackFram/llama-68m`` when the family is unknown.
        """
        model_type = None
        name_or_path = None
        if isinstance(target, str):
            name_or_path = target
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(target, trust_remote_code=True)
                model_type = getattr(cfg, "model_type", None)
            except Exception as e:
                logger.warning(
                    f"[DRAFTER_SELECT] could not read config for target={target!r}: {e}; "
                    f"falling back to default drafter"
                )
        elif hasattr(target, "model_type"):
            model_type = target.model_type
            name_or_path = getattr(target, "name_or_path", None) or getattr(target, "_name_or_path", None)
        elif hasattr(target, "config"):
            model_type = getattr(target.config, "model_type", None)
            name_or_path = getattr(target.config, "name_or_path", None) or getattr(target.config, "_name_or_path", None)

        drafter_id, source = select_drafter_for_target(
            target_model_type=model_type, target_name_or_path=name_or_path,
        )
        logger.info(
            f"[DRAFTER_SELECT] target_model_type={model_type!r} "
            f"→ drafter={drafter_id!r} (source={source})"
        )
        return cls(ssm_model_name=drafter_id, num_workers=num_workers, device=device)
    
    def build_trees_parallel(
        self,
        input_ids: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        beam_width: Union[int, Sequence[int]],
        max_depth: int,
        *,
        tree_budget: Optional[int] = None,
        tree_min_log_prob: Optional[float] = None,
    ) -> List:
        # Tree-shape policies, in precedence order:
        #
        # 1. Sequoia (https://arxiv.org/abs/2402.12374): if ``beam_width`` is a
        #    list/tuple, it's interpreted as the Sequoia static per-depth
        #    widths plan. Depth ``d`` expands each frontier node to
        #    ``beam_width[d]`` children. No per-step ranking cost.
        # 2. EAGLE-2 (https://arxiv.org/abs/2406.16858): if ``tree_budget`` or
        #    ``tree_min_log_prob`` is set, expand a uniform (depth × beam_width)
        #    grid and then prune via ``budgeted_expand_plan``.
        # 3. Default: full (depth × beam_width) grid — byte-identical to the
        #    pre-EAGLE-2 behavior.
        batch_size = input_ids.shape[0]
        chunk_size = (batch_size + self.num_workers - 1) // self.num_workers

        all_results = [None] * batch_size

        def worker_fn(worker_idx: int, batch_indices: List[int]):
            ssm = self.ssms[worker_idx]
            stream = self.streams[worker_idx]

            with torch.cuda.stream(stream):
                results = self._build_trees_batched(
                    batch_indices, input_ids, seq_lengths, ssm, beam_width, max_depth,
                    tree_budget=tree_budget, tree_min_log_prob=tree_min_log_prob,
                )
                for batch_idx, tree in results:
                    all_results[batch_idx] = tree
        
        # 启动线程
        threads = []
        for worker_idx in range(self.num_workers):
            start = worker_idx * chunk_size
            end = min(start + chunk_size, batch_size)
            if start < batch_size:
                t = threading.Thread(target=worker_fn, args=(worker_idx, list(range(start, end))))
                threads.append(t)
                t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 同步所有 streams
        for stream in self.streams:
            stream.synchronize()
        
        return all_results

    def _build_trees_batched(
        self,
        batch_indices: List[int],
        input_ids: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        ssm,
        beam_width: Union[int, Sequence[int]],
        max_depth: int,
        *,
        tree_budget: Optional[int] = None,
        tree_min_log_prob: Optional[float] = None,
    ) -> List:

        # Sequoia per-depth plan: list/tuple overrides scalar beam_width and
        # caps tree depth. Each depth expands to ``width_plan[depth]`` children.
        if isinstance(beam_width, (list, tuple)):
            width_plan: Optional[List[int]] = [int(w) for w in beam_width]
            max_depth = min(max_depth, len(width_plan))
        else:
            width_plan = None

        # Some drafter repos (e.g. JackFram/llama-160m) don't set
        # pad_token_id in their config, leaving it None. Fall back to
        # eos_token_id, then bos_token_id, then 0 so downstream
        # torch.full() calls don't trip on a None fill_value.
        pad_token_id = (
            getattr(ssm.config, 'pad_token_id', None)
            or getattr(ssm.config, 'eos_token_id', None)
            or getattr(ssm.config, 'bos_token_id', None)
            or 0
        )
        if isinstance(pad_token_id, (list, tuple)):
            pad_token_id = int(pad_token_id[0])
        pad_token_id = int(pad_token_id)
        
        trees = {}
        valid_inputs = {}
        prefix_lengths = {}
        t0 = time.perf_counter()
        
        for batch_idx in batch_indices:
            actual_len = seq_lengths[batch_idx].item()
            valid_input_ids = input_ids[batch_idx, :actual_len]
            valid_inputs[batch_idx] = valid_input_ids
            prefix_lengths[batch_idx] = max(actual_len - 1, 0)
            
            root_token = valid_input_ids[-1].item()
            trees[batch_idx] = SpeculativeTree(root_token, f"req_{batch_idx}")
        
        # ========== 预计算 prefix cache ==========
        max_prefix_len = max(prefix_lengths.values())
        
        if max_prefix_len == 0:
            prefix_cache = None
        else:
            padded_prefixes = []
            prefix_masks = []
            
            for batch_idx in batch_indices:
                pf_len = prefix_lengths[batch_idx]
                
                if pf_len > 0:
                    prefix = valid_inputs[batch_idx][:-1]
                else:
                    prefix = torch.tensor([], dtype=torch.long, device=self.device)
                
                pad_len = max_prefix_len - pf_len
                
                if pf_len > 0:
                    padded_prefixes.append(torch.cat([
                        torch.full((pad_len,), pad_token_id, dtype=torch.long, device=self.device),
                        prefix
                    ]))
                else:
                    padded_prefixes.append(
                        torch.full((max_prefix_len,), pad_token_id, dtype=torch.long, device=self.device)
                    )
                
                prefix_masks.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=self.device),
                    torch.ones(pf_len, dtype=torch.long, device=self.device)
                ]))
            
            batch_prefixes = torch.stack(padded_prefixes)
            batch_prefix_masks = torch.stack(prefix_masks)
            
            with torch.no_grad():
                prefix_outputs = ssm(batch_prefixes, attention_mask=batch_prefix_masks, use_cache=True)
                prefix_cache = prefix_outputs.past_key_values
        
        idx_map = {batch_idx: i for i, batch_idx in enumerate(batch_indices)}
        
        # ========== 按 depth 扩展 ==========
        t1 = time.perf_counter()
        # logger.info(f"Prefix processing time: {t1 - t0:.4f}s")
        
        t_forward = 0
        t_postprocess = 0
        t_prepare = 0
        for depth in range(max_depth):
            
            t0 = time.perf_counter()
            all_paths = []
            node_mapping = []
            cache_indices = []
            
            for batch_idx in batch_indices:
                tree = trees[batch_idx]
                root_token = valid_inputs[batch_idx][-1].item()
                
                for node in tree.get_nodes_at_depth(depth):
                    path = node.get_path_from_root()
                    path_tokens = torch.tensor([root_token] + path, dtype=torch.long, device=self.device)
                    all_paths.append(path_tokens)
                    node_mapping.append((batch_idx, node))
                    cache_indices.append(idx_map[batch_idx])
            
            if not all_paths:
                break
            
            num_nodes = len(all_paths)
            max_path_len = max(len(p) for p in all_paths)
            max_pf_len = max(prefix_lengths[nm[0]] for nm in node_mapping)
            total_mask_len = max_pf_len + max_path_len
            
            # 预分配
            batch_paths = torch.full((num_nodes, max_path_len), pad_token_id, dtype=torch.long, device=self.device)
            batch_path_masks = torch.zeros((num_nodes, total_mask_len), dtype=torch.long, device=self.device)
            
            # 填充
            for i, path in enumerate(all_paths):
                path_len = len(path)
                batch_paths[i, -path_len:] = path
                
                batch_idx = node_mapping[i][0]
                pf_len = prefix_lengths[batch_idx]
                prefix_pad_len = max_pf_len - pf_len
                
                batch_path_masks[i, prefix_pad_len:prefix_pad_len + pf_len] = 1
                batch_path_masks[i, -path_len:] = 1
            
            # cache
            if prefix_cache is not None:
                node_cache = DynamicCache()
                # transformers 5.x renamed tuple indexing into layers[i].keys/values.
                # Fall back to the legacy subscript API when present.
                if hasattr(prefix_cache, "layers"):
                    for layer_idx, layer in enumerate(prefix_cache.layers):
                        node_cache.update(layer.keys[cache_indices], layer.values[cache_indices], layer_idx)
                else:
                    for layer_idx in range(len(prefix_cache)):
                        key, value = prefix_cache[layer_idx]
                        node_cache.update(key[cache_indices], value[cache_indices], layer_idx)
            else:
                node_cache = None
            
            t_prepare += time.perf_counter() - t0
            t0 = time.perf_counter()
            # forward
            with torch.no_grad():
                outputs = ssm(batch_paths, attention_mask=batch_path_masks, past_key_values=node_cache, use_cache=False)
                all_logits = outputs.logits[:, -1, :]
            
            t_forward += time.perf_counter() - t0
            t0 =    time.perf_counter()
            # Per-depth width: Sequoia plan overrides scalar beam_width.
            if width_plan is not None:
                depth_width = width_plan[depth] if depth < len(width_plan) else 0
            else:
                depth_width = int(beam_width)
            if depth_width <= 0:
                break
            # 批量 topk
            _, all_top_k_indices = torch.topk(all_logits, k=depth_width, dim=-1)
            all_probs = torch.softmax(all_logits, dim=-1)
            all_top_k_probs = torch.gather(all_probs, 1, all_top_k_indices)

            all_top_k_indices_np = all_top_k_indices.cpu().numpy()
            all_top_k_probs_np = all_top_k_probs.cpu().numpy()

            batch_node_results = {}
            for i, (batch_idx, node) in enumerate(node_mapping):
                candidates = [(int(all_top_k_indices_np[i, j]), float(all_top_k_probs_np[i, j]))
                            for j in range(depth_width)]
                if batch_idx not in batch_node_results:
                    batch_node_results[batch_idx] = []
                batch_node_results[batch_idx].append((node, candidates))

            # EAGLE-2 (arXiv 2406.16858) tree shaping, applied per-batch so one
            # sample's overly greedy expansion can't starve another sample in
            # the same drafter call. When both tree_budget and
            # tree_min_log_prob are None this loop is a no-op that rewrites
            # the candidate lists back to their original form.
            if tree_budget is not None or tree_min_log_prob is not None:
                for batch_idx, per_node in batch_node_results.items():
                    frontier: List[FrontierNode] = []
                    for node, candidates in per_node:
                        parent_log = _node_path_log_prob(node)
                        for cand_token, cand_prob in candidates:
                            cand_log = parent_log + (math.log(cand_prob) if cand_prob > 0 else float("-inf"))
                            # Identity key uniquely identifies one candidate edge.
                            handle = (id(node), int(cand_token), float(cand_prob))
                            frontier.append(
                                FrontierNode(
                                    node_handle=handle,
                                    path_log_prob=cand_log,
                                    depth=depth + 1,
                                )
                            )
                    survivors = budgeted_expand_plan(
                        frontier,
                        budget=tree_budget if tree_budget is not None else len(frontier),
                        min_log_prob=tree_min_log_prob,
                    )
                    kept_handles = {fn.node_handle for fn in survivors}
                    filtered_per_node = []
                    for node, candidates in per_node:
                        kept = [
                            (t, p)
                            for t, p in candidates
                            if (id(node), int(t), float(p)) in kept_handles
                        ]
                        if kept:
                            filtered_per_node.append((node, kept))
                    batch_node_results[batch_idx] = filtered_per_node

            any_new = False
            for batch_idx in batch_indices:
                if batch_idx not in batch_node_results:
                    continue
                tree = trees[batch_idx]
                nodes = [nc[0] for nc in batch_node_results[batch_idx]]
                candidates = [nc[1] for nc in batch_node_results[batch_idx]]
                try:
                    if tree.add_layer(nodes, candidates):
                        any_new = True
                except ValueError:
                    pass
            t_postprocess += time.perf_counter() - t0
            
            if not any_new:
                break
        
        # logger.info(f"forward: {t_forward:.4f}s, postprocess: {t_postprocess:.4f}s, total prepare: {t_prepare:.4f}s")
        return [(idx, trees[idx]) for idx in batch_indices]