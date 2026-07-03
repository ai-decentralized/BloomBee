"""EAGLE-2 drafter for BloomBee.

Implements the EAGLE-2 algorithm (Li et al., arXiv 2406.16858) directly inside
BloomBee. The drafter:

* Takes the **target model's last-layer hidden state** for the last committed
  token plus the token id, and produces draft-token candidates.
* Uses a single Llama decoder layer with an `fc(concat(emb, hidden))` projection
  on the input — this is the EAGLE head architecture from
  ``eagle/model/cnets.py:Model``.
* Loads a pre-trained head selected for the target LLaMA family (or any
  user-provided HF checkpoint with the same layout).
* Implements the EAGLE-2 dynamic tree growth: per-iteration top-k expansion
  across the latest layer plus global top-m reranking by accumulated path
  log-probability.

Why we re-implement the dynamic tree instead of calling the upstream
``eagle.model.ea_model.EaModel.eagenerate``:

* The pip-distributed ``eagle-llm`` package only ships EAGLE-1 (static
  ``mc_sim_7b_63`` 63-node tree, batch=1 only).
* EAGLE-2's win comes from the dynamic tree, which is at most ~80 lines of
  Python on top of the EAGLE head.
* Re-implementing keeps batch>1 inference working in BloomBee and lets us
  thread the ``[B, H]`` hidden tensor without reshape gymnastics.

Output contract matches ``MultiSSMDrafter.build_trees_parallel``:

    drafter.build_trees_parallel(
        input_ids: torch.LongTensor,           # [B, L_max] target prefix
        seq_lengths: torch.LongTensor,         # [B] real lengths
        beam_width: int | Sequence[int],       # ignored if total_token set
        max_depth: int,
        *,
        prev_last_hidden: torch.FloatTensor,   # [B, H]  EAGLE conditioning
        prev_last_token: torch.LongTensor,     # [B]     EAGLE conditioning
        total_token: int | None = None,        # EAGLE-2 global budget incl. root
        topk_per_step: int = 10,               # EAGLE-2 per-layer top-k
        max_new_layers: int = 6,               # depth cap (paper uses ~6)
        ...
    ) -> List[SpeculativeTree]

This signature is a strict superset of ``MultiSSMDrafter.build_trees_parallel``
plus EAGLE-conditioning kwargs that ``_sample_with_session`` adds when they
are available.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.utils.logging import get_logger
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from bloombee.models.llama.spe_dec_tree import SpeculativeTree

logger = get_logger()


_EAGLE_DRAFTER_ENV = "BLOOMBEE_EAGLE_DRAFTER"
_EAGLE_TREE_BUDGET_ENV = "BLOOMBEE_EAGLE_TREE_BUDGET"
_EAGLE_TOPK_PER_STEP_ENV = "BLOOMBEE_EAGLE_TOPK_PER_STEP"
_EAGLE_DEPTH_ENV = "BLOOMBEE_EAGLE_DEPTH"
# Bandwidth-adaptive tree budget: when the operator declares the S2S link
# bandwidth (Mbps) via this env var, the drafter shrinks the tree on slow links.
# Rationale: the speculative S2S hidden payload scales as B*(budget+1)*H*bytes, so
# on a bandwidth-starved link a smaller tree (fewer nodes to ship) wins on
# throughput even though it accepts slightly fewer tokens/round. Empirically
# (vicuna-13b, 2-server split, batch 32, E5=20Mbps): budget 4 -> 1.26x no-SD,
# budget 8 -> 1.23x, budget 12 -> 1.06x, budget 16 -> 0.92x, budget 24 -> 0.76x;
# i.e. budget 4-8 is the sweet spot at 20Mbps, while on a fast link a larger tree
# (higher accept) wins. See results/.../budget_sweep_b32_e5 and CODEX_VLLM_ANALYSIS.
_EAGLE_BANDWIDTH_MBPS_ENV = "BLOOMBEE_EAGLE_BANDWIDTH_MBPS"

def select_bandwidth_adaptive_budget(bandwidth_mbps: Optional[float], default_budget: int) -> int:
    """Pick an EAGLE tree budget for the given S2S link bandwidth (Mbps).

    Returns ``default_budget`` when bandwidth is unknown (None / <= 0) so the
    behavior is unchanged unless the operator opts in. The thresholds below are
    calibrated from the batch-32 E1-E5 throughput sweep: small trees win on slow
    links (payload-bound), large trees win on fast links (accept-bound)."""
    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        return default_budget
    # Cap by the operator's default so this only ever *shrinks* the tree on slow
    # links relative to what they asked for (never silently grows it).
    if bandwidth_mbps <= 30:        # ~E5 (20 Mbps): heavily payload-bound
        chosen = 6
    elif bandwidth_mbps <= 150:     # ~E4 (125 Mbps)
        chosen = 8
    elif bandwidth_mbps <= 400:     # ~E3 (250 Mbps)
        chosen = 10
    else:                            # ~E1/E2 (>=500 Mbps / LAN): accept-bound
        chosen = default_budget
    return max(1, min(int(chosen), int(default_budget)))

_EAGLE_DRAFTER_REGISTRY: Dict[str, Dict[int, str]] = {
    # Official yuhuili EAGLE/EAGLE-2-compatible head checkpoints. Keep this
    # conservative: do not cross-use Llama-2, Vicuna, and Llama-3 heads just
    # because they share `model_type="llama"`.
    "llama2-chat": {
        4096: "yuhuili/EAGLE-llama2-chat-7B",
        5120: "yuhuili/EAGLE-llama2-chat-13B",
        8192: "yuhuili/EAGLE-llama2-chat-70B",
    },
    "vicuna": {
        4096: "yuhuili/EAGLE-Vicuna-7B-v1.3",
        5120: "yuhuili/EAGLE-Vicuna-13B-v1.3",
        6656: "yuhuili/EAGLE-Vicuna-33B-v1.3",
    },
    "llama3": {
        4096: "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        8192: "yuhuili/EAGLE-LLaMA3-Instruct-70B",
    },
    "llama3.1": {
        4096: "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    },
}


def _config_name_candidates(target_config: object, target_name_or_path: Optional[str]) -> List[str]:
    names: List[str] = []
    if target_name_or_path:
        names.append(str(target_name_or_path))
    for attr in ("name_or_path", "_name_or_path", "name"):
        value = getattr(target_config, attr, None)
        if value and isinstance(value, str):
            names.append(value)
    return names


def _infer_eagle_family(name_candidates: Sequence[str]) -> Optional[str]:
    joined = " ".join(name_candidates).lower()
    if "vicuna" in joined:
        return "vicuna"
    if "llama-3.1" in joined or "llama3.1" in joined:
        return "llama3.1"
    if "llama-3" in joined or "llama3" in joined:
        return "llama3"
    if ("llama-2" in joined or "llama2" in joined) and "chat" in joined:
        return "llama2-chat"
    return None


def select_eagle_drafter_for_target(
    target_config: object,
    target_name_or_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Select an EAGLE head checkpoint for a LLaMA-family target.

    Returns ``(repo_id, source)`` where source is ``env`` or
    ``registry:<family>``. Unknown targets fail closed instead of silently
    borrowing a mismatched drafter checkpoint.
    """
    env_override = os.environ.get(_EAGLE_DRAFTER_ENV)
    if env_override:
        return env_override, "env"

    model_type = (getattr(target_config, "model_type", None) or "").lower()
    if model_type and model_type != "llama":
        raise ValueError(
            "EAGLE drafter auto-selection only supports LLaMA-family targets; "
            "pass ea_model_path explicitly for custom compatible checkpoints."
        )

    names = _config_name_candidates(target_config, target_name_or_path)
    family = _infer_eagle_family(names)
    if family is None:
        raise ValueError(
            "Could not infer a compatible EAGLE drafter for this LLaMA target. "
            "Pass ea_model_path explicitly, or set BLOOMBEE_EAGLE_DRAFTER."
        )

    hidden_size = int(getattr(target_config, "hidden_size", 0) or 0)
    family_registry = _EAGLE_DRAFTER_REGISTRY[family]
    if hidden_size not in family_registry:
        raise ValueError(
            f"No registered EAGLE drafter for family={family!r}, hidden_size={hidden_size}. "
            "Pass ea_model_path explicitly for a compatible checkpoint."
        )
    return family_registry[hidden_size], f"registry:{family}"


def _build_eagle_head(
    target_hidden_size: int,
    target_vocab_size: int,
    eagle_config: Optional[object] = None,
    target_config: Optional[object] = None,
):
    """Construct the EAGLE drafter head, faithful to ``eagle.model.cnets.Model``.

    Differences from a stock HF Llama decoder block — the trap that made our
    first port silently collapse to a single token in autoregressive replay:

    * EAGLE's custom ``LlamaDecoderLayer.forward`` SKIPS ``input_layernorm`` on
      the first (and only) layer (``if self.index != 0:`` guard). HF's stock
      ``LlamaDecoderLayer`` always applies it. The skip matters because the
      ``fc(concat(emb, hidden))`` output has carefully-tuned magnitudes that
      RMSNorm destroys.
    * EAGLE has no final ``self.norm`` on top of the layer output; the
      drafter hidden goes straight to ``lm_head`` (target's, shared).

    transformers >= 5 moved RoPE to the model level — we keep our own
    ``LlamaRotaryEmbedding`` and run the attention path manually so we can
    skip ``input_layernorm`` exactly the way the EAGLE checkpoint expects.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
    )

    def _cfg_value(name: str, default):
        value = getattr(eagle_config, name, None)
        if value is None and target_config is not None:
            value = getattr(target_config, name, None)
        return default if value is None else value

    if target_hidden_size == 4096:
        default_intermediate, default_heads = 11008, 32
    elif target_hidden_size == 5120:
        default_intermediate, default_heads = 13824, 40
    elif target_hidden_size == 6656:
        default_intermediate, default_heads = 17920, 52
    elif target_hidden_size == 8192:
        default_intermediate, default_heads = 28672, 64
    else:
        default_intermediate = int(target_hidden_size * 8 / 3)
        default_heads = max(1, target_hidden_size // 128)

    num_attention_heads = int(_cfg_value("num_attention_heads", default_heads))
    num_key_value_heads = int(_cfg_value("num_key_value_heads", num_attention_heads))

    cfg = LlamaConfig(
        hidden_size=target_hidden_size,
        intermediate_size=int(_cfg_value("intermediate_size", default_intermediate)),
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=target_vocab_size,
        max_position_embeddings=int(_cfg_value("max_position_embeddings", 4096)),
        rms_norm_eps=float(_cfg_value("rms_norm_eps", 1e-6)),
        rope_scaling=_cfg_value("rope_scaling", None),
        rope_theta=float(_cfg_value("rope_theta", 10000.0)),
        hidden_act=str(_cfg_value("hidden_act", "silu")),
        pretraining_tp=int(_cfg_value("pretraining_tp", 1)),
    )

    class EAGLELayer(nn.Module):
        """Faithful port of EAGLE's custom decoder block (cnets.LlamaDecoderLayer
        with index=0 — i.e. no input_layernorm).

        State_dict keys mirror the EAGLE checkpoint:
          self_attn.{q,k,v,o}_proj
          mlp.{gate,up,down}_proj
          post_attention_layernorm
        """

        def __init__(self):
            super().__init__()
            try:
                self.self_attn = LlamaAttention(config=cfg, layer_idx=0)
            except TypeError:
                self.self_attn = LlamaAttention(config=cfg)
            self.mlp = LlamaMLP(cfg)
            self.post_attention_layernorm = LlamaRMSNorm(target_hidden_size, eps=cfg.rms_norm_eps)

        def forward(self, hidden_states, attention_mask, position_ids, position_embeddings, past_key_values):
            residual = hidden_states
            # NOTE: NO input_layernorm here — EAGLE's index==0 block skips it.
            attn_kwargs = dict(
                position_ids=position_ids,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
            try:
                attn_out = self.self_attn(hidden_states=hidden_states, **attn_kwargs)
            except TypeError:
                attn_out = self.self_attn(hidden_states, **attn_kwargs)
            if isinstance(attn_out, tuple):
                attn_h = attn_out[0]
            else:
                attn_h = attn_out
            hidden_states = residual + attn_h

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, past_key_values

    class EAGLEHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(target_vocab_size, target_hidden_size)
            self.fc = nn.Linear(2 * target_hidden_size, target_hidden_size, bias=True)
            self.layer = EAGLELayer()
            try:
                self.rotary = LlamaRotaryEmbedding(config=cfg)
            except TypeError:
                self.rotary = LlamaRotaryEmbedding(
                    cfg.hidden_size // cfg.num_attention_heads,
                    max_position_embeddings=cfg.max_position_embeddings,
                )

        def forward(
            self,
            hidden_states: torch.Tensor,   # [B, S, H] target last-layer hidden (or drafter's previous hidden)
            input_ids: torch.LongTensor,   # [B, S] tokens to embed for fc
            position_ids: torch.LongTensor,
            past_key_values: Optional[DynamicCache] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, DynamicCache]:
            inputs_embeds = self.embed_tokens(input_ids).to(hidden_states.dtype)
            # EAGLE's fc(concat([emb, hidden])) → 4096; this matches the
            # checkpoint's `fc.weight` shape (4096, 8192). Order: emb first.
            x = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))
            cos, sin = self.rotary(x, position_ids)

            if past_key_values is None:
                past_key_values = DynamicCache(config=cfg)

            if attention_mask is None:
                # 4D additive causal mask. With a non-empty cache, every query
                # can see all cached keys plus the causal prefix inside this
                # chunk.
                B, q_len, _ = x.shape
                past_len = int(past_key_values.get_seq_length(0))
                kv_len = past_len + q_len
                neg_inf = torch.finfo(x.dtype).min
                q_idx = torch.arange(q_len, device=x.device).view(-1, 1)
                k_idx = torch.arange(kv_len, device=x.device).view(1, -1)
                allow = (k_idx < past_len) | ((k_idx - past_len) <= q_idx)
                additive = torch.zeros((q_len, kv_len), dtype=x.dtype, device=x.device)
                additive = additive.masked_fill(~allow, neg_inf)
                additive = additive.view(1, 1, q_len, kv_len).expand(B, 1, q_len, kv_len)
            else:
                additive = attention_mask.to(device=x.device, dtype=x.dtype)

            new_hidden, past_key_values = self.layer(
                hidden_states=x,
                attention_mask=additive,
                position_ids=position_ids,
                position_embeddings=(cos, sin),
                past_key_values=past_key_values,
            )
            return new_hidden, past_key_values

    return EAGLEHead(), cfg


import itertools as _itertools
_CAND_NODE_COUNTER = _itertools.count()


@dataclass
class _CandNode:
    """Mutable EAGLE-tree node used during expansion."""
    token_id: int
    parent: Optional["_CandNode"]
    depth: int
    log_p: float                # log p(this_token | path_so_far) at draft head
    path_log_p: float           # cumulative path log-prob (sum log_p's)
    hidden: Optional[torch.Tensor] = None  # drafter hidden for this token
    expanded: bool = False
    children: List["_CandNode"] = field(default_factory=list)
    # Position assigned during flat-tree linearization (set later)
    flat_index: Optional[int] = None
    # Monotonic allocation index, assigned at construction. Makes sibling order
    # deterministic and tensor-reproducible (replaces id()-based tie-breaks in
    # _bind_into_speculative_tree). Equivalent to the old id() ordering because
    # nodes are created in the same deterministic loop order; a counter is a
    # language-level allocation order that a tensor path can reproduce.
    creation_index: int = field(default_factory=lambda: next(_CAND_NODE_COUNTER))


@dataclass
class _PrefixCacheState:
    """Per-batch-row shifted-prefix cache for EAGLE replay."""
    cache: DynamicCache
    cache_len: int = 0
    cache_ids: Optional[torch.LongTensor] = None
    last_hidden: Optional[torch.Tensor] = None


@dataclass
class _TopKLogProbs:
    values: torch.Tensor
    indices: torch.LongTensor


@dataclass
class _PrefixBuildJob:
    batch_index: int
    root_token: int
    root_hidden: torch.Tensor
    prefix_cache: DynamicCache
    prefix_next_pos: int


def _topk_per_layer_indices(layer_nodes: List[_CandNode], k: int) -> List[_CandNode]:
    """Pick top-k nodes from a single tree depth by cumulative path_log_p."""
    if k <= 0 or not layer_nodes:
        return []
    if len(layer_nodes) <= k:
        return list(layer_nodes)
    # Stable: ties -> insertion order
    ranked = sorted(enumerate(layer_nodes), key=lambda iv: (-iv[1].path_log_p, iv[0]))
    return [n for _, n in ranked[:k]]


def _topm_global(all_nodes: List[_CandNode], m: int) -> List[_CandNode]:
    """Pick top-m nodes across the whole tree (root excluded), tie-break:
    shallower first to keep the tree connected."""
    if m <= 0 or not all_nodes:
        return []
    if len(all_nodes) <= m:
        return list(all_nodes)
    ranked = sorted(
        all_nodes,
        key=lambda n: (-n.path_log_p, n.depth),
    )
    return ranked[:m]


def _close_under_parents(selected: List[_CandNode]) -> List[_CandNode]:
    """Add ancestors of each selected node so the resulting set is parent-closed
    (a connected sub-tree rooted at the original root).

    The caller's root node is implicit (corresponds to BloomBee's
    ``SpeculativeTree.root``); we exclude it from the output. ``selected`` is
    expected to contain only non-root candidates.
    """
    chosen = set()
    out: List[_CandNode] = []
    stack = list(selected)
    while stack:
        n = stack.pop()
        if id(n) in chosen:
            continue
        chosen.add(id(n))
        if n.parent is None:
            # This is the synthetic root — never include in output.
            continue
        out.append(n)
        if id(n.parent) not in chosen:
            stack.append(n.parent)
    # Sort by depth so prepare_incremental_tree_batch sees a parent-before-child
    # order (every node's parent is already grafted by the time it lands).
    out.sort(key=lambda n: n.depth)
    return out


def _bind_into_speculative_tree(root_token: int, kept: List[_CandNode]) -> SpeculativeTree:
    """Materialize an EAGLE-2 candidate set into a BloomBee SpeculativeTree.

    BloomBee's ``SpeculativeTree`` exposes the root as a TreeNode and each
    TreeNode has ``add_child(token_id, probability)``. We replay the parent-
    closed kept set in depth order and update ``total_nodes`` / ``max_depth``
    so that downstream code (``prepare_incremental_tree_batch``) sees a
    well-formed tree.
    """
    tree = SpeculativeTree(root_token, request_id="eagle2_req")
    by_id: dict[int, object] = {}
    # Map each EAGLE _CandNode (id(...)) to the corresponding BloomBee TreeNode.
    # Root: kept doesn't include root explicitly, so when n.parent is None we
    # graft onto tree.root.
    kept_sorted = sorted(kept, key=lambda n: (n.depth, n.creation_index))
    for n in kept_sorted:
        if n.parent is None:
            parent_bb = tree.root
        else:
            parent_bb = by_id.get(id(n.parent), tree.root)
        prob = math.exp(max(n.log_p, -20.0))
        bb_node = parent_bb.add_child(n.token_id, prob)
        by_id[id(n)] = bb_node
        if bb_node.depth > tree.max_depth:
            tree.max_depth = bb_node.depth
        tree.total_nodes += 1
    return tree


def _default_draft_token_budget(beam_width: Union[int, Sequence[int]], max_depth: int) -> int:
    """Return BloomBee's requested number of draft nodes, excluding the root."""
    depth = max(0, int(max_depth))
    if isinstance(beam_width, (list, tuple)):
        total = 0
        running = 1
        for width in beam_width[:depth]:
            running *= max(int(width), 1)
            total += running
        return total
    return depth * max(int(beam_width), 1)


def _eagle2_max_candidate_depth(max_depth: int, *, explicit_tree_budget: bool) -> int:
    """Translate BloomBee depth to EAGLE-2's official dynamic-tree depth.

    BloomBee's legacy ``max_tree_depth`` is the maximum draft path length. The
    EAGLE-2 reference ``depth`` flag is one less than that: it emits the first
    draft layer before the loop, then runs ``range(depth)``. With a paper-style
    tree budget we use the official default depth=5, i.e. up to 6 draft tokens.
    """
    max_candidate_depth = max(1, int(max_depth))
    if explicit_tree_budget:
        official_depth = max(1, int(os.environ.get(_EAGLE_DEPTH_ENV, "5")))
        max_candidate_depth = max(max_candidate_depth, official_depth + 1)
    return max_candidate_depth


def _default_eagle_topk_per_step(total_token: int, *, explicit_tree_budget: bool, do_sample: bool) -> int:
    env_value = os.environ.get(_EAGLE_TOPK_PER_STEP_ENV)
    if env_value is not None and env_value.strip():
        return max(1, int(env_value))

    # The paper tree uses total_token=60 and top-k=10. For BloomBee's default
    # "draft 5 tokens" workload, a compact EAGLE-2 tree keeps the high-accept
    # dynamic-tree behavior while avoiding the 10x10 frontier work that made
    # drafter construction dominate latency.
    if explicit_tree_budget and not bool(do_sample) and int(total_token) <= 11:
        return 3
    if explicit_tree_budget and not bool(do_sample) and int(total_token) <= 21:
        return 5
    return 10


class EAGLEDrafter:
    """EAGLE-2 drafter for BloomBee (drop-in replacement for MultiSSMDrafter)."""

    def __init__(
        self,
        ea_model_path: str,
        target_hidden_size: int,
        target_vocab_size: int,
        target_lm_head: torch.nn.Module,
        target_config: Optional[object] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        try:
            eagle_cfg = AutoConfig.from_pretrained(ea_model_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(
                "[EAGLEDrafter] could not load config for %s (%s); "
                "falling back to target model architecture",
                ea_model_path,
                e,
            )
            eagle_cfg = None
        self.head, self.head_cfg = _build_eagle_head(
            target_hidden_size,
            target_vocab_size,
            eagle_config=eagle_cfg,
            target_config=target_config,
        )
        self.head = self.head.to(self.device).to(self.dtype).eval()
        self.target_lm_head = target_lm_head  # kept for metadata/debugging only
        if not hasattr(target_lm_head, "weight") or target_lm_head.weight is None:
            raise ValueError("EAGLEDrafter requires a target lm_head with a weight tensor")
        # The distributed BloomBee client sometimes keeps lm_head on CPU. If
        # the target head is already on the drafter device, reuse its storage
        # instead of making a second 400MB+ copy for 33B-class models.
        target_weight = target_lm_head.weight.detach()
        if (
            target_weight.device == self.device
            and target_weight.dtype == self.dtype
            and target_weight.is_contiguous()
        ):
            self.lm_head_weight = target_weight
        else:
            self.lm_head_weight = target_weight.to(
                device=self.device,
                dtype=self.dtype,
            ).contiguous()
        bias = getattr(target_lm_head, "bias", None)
        if torch.is_tensor(bias):
            target_bias = bias.detach()
            if (
                target_bias.device == self.device
                and target_bias.dtype == self.dtype
                and target_bias.is_contiguous()
            ):
                self.lm_head_bias = target_bias
            else:
                self.lm_head_bias = target_bias.to(device=self.device, dtype=self.dtype).contiguous()
        else:
            self.lm_head_bias = None
        self.uses_eagle_hidden_states = True
        # AutoDistributedSpeculativeModel.generate uses this when callers do
        # not pass tree_budget. BloomBee's runtime default is tuned for the
        # product metric "draft 5 tokens, accept as many as possible": a small
        # dynamic tree preserves EAGLE-2 acceptance while keeping verify latency
        # under control. Set BLOOMBEE_EAGLE_TREE_BUDGET=59 for the paper tree.
        self.default_tree_budget = max(1, int(os.environ.get(_EAGLE_TREE_BUDGET_ENV, "10")))
        # Optional bandwidth-adaptive budget: shrink the tree on slow S2S links so
        # the per-round speculative payload (B*(budget+1)*H) stays under the link's
        # throughput break-even. Only applies when the operator sets the bandwidth
        # hint AND does not pass an explicit tree_budget at generate() time.
        _bw_env = os.environ.get(_EAGLE_BANDWIDTH_MBPS_ENV)
        if _bw_env is not None:
            try:
                _bw = float(_bw_env)
            except ValueError:
                _bw = None
            adapted = select_bandwidth_adaptive_budget(_bw, self.default_tree_budget)
            if adapted != self.default_tree_budget:
                logger.info(
                    "EAGLE bandwidth-adaptive budget: link=%s Mbps -> tree_budget %d (was %d)",
                    _bw_env, adapted, self.default_tree_budget,
                )
            self.default_tree_budget = adapted
        self._prefix_states: Dict[int, _PrefixCacheState] = {}

        self._load_eagle_weights(ea_model_path)

    def reorder_prefix_states(self, perm: Sequence[int]) -> None:
        """Remap per-row drafter prefix caches after active-row compaction.

        ``perm`` lists the surviving original row indices in their new order
        (new row i held what was row perm[i]). Rows not in ``perm`` are
        finished; their cached state is dropped. Without this remap the next
        build would validate the wrong row's cache and fall back to a full
        (byte-identical but slow) prefix replay for every compacted row.
        """
        old = self._prefix_states
        self._prefix_states = {}
        for new_idx, old_idx in enumerate(perm):
            state = old.get(int(old_idx))
            if state is not None:
                self._prefix_states[new_idx] = state

    def _load_eagle_weights(self, path: str) -> None:
        """Load yuhuili-style EAGLE weights into ``self.head``.

        The official EAGLE checkpoint stores `embed_tokens.weight`,
        `fc.{weight,bias}`, and `layers.0.*` (one decoder layer). We map those
        onto our `EAGLEHead` (which has `embed_tokens`, `fc`, `layer`) and load
        as a missing-key-tolerant state_dict.
        """
        if os.path.isdir(path):
            repo_path = path
        else:
            from huggingface_hub import snapshot_download

            repo_path = snapshot_download(
                repo_id=path,
                allow_patterns=["*.bin", "*.safetensors", "config.json"],
            )
        sd: dict = {}
        # Try safetensors then .bin
        candidates = []
        for f in os.listdir(repo_path):
            if f.endswith(".safetensors"):
                candidates.append(os.path.join(repo_path, f))
        if not candidates:
            for f in os.listdir(repo_path):
                if f.endswith(".bin"):
                    candidates.append(os.path.join(repo_path, f))
        if not candidates:
            raise FileNotFoundError(f"No EAGLE weight files in {repo_path}")

        for fpath in candidates:
            if fpath.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd.update(load_file(fpath))
            else:
                sd.update(torch.load(fpath, map_location="cpu", weights_only=False))

        # EAGLE-1 ckpt typically uses keys like:
        #   'embed_tokens.weight'
        #   'fc.weight', 'fc.bias'
        #   'layers.0.<...>'
        # Our EAGLEHead exposes:
        #   'embed_tokens.weight', 'fc.*', 'layer.<...>'
        # Map 'layers.0.' -> 'layer.'
        remapped = {}
        for k, v in sd.items():
            if k.startswith("layers.0."):
                remapped[k.replace("layers.0.", "layer.", 1)] = v
            else:
                remapped[k] = v

        missing, unexpected = self.head.load_state_dict(remapped, strict=False)
        logger.info(
            "[EAGLEDrafter] loaded %s. missing=%d unexpected=%d",
            path, len(missing), len(unexpected),
        )
        if unexpected:
            logger.info("[EAGLEDrafter] unexpected keys (sample): %s", unexpected[:5])

    @classmethod
    def for_target(
        cls,
        target_model,                                  # bloombee DistributedLlamaForCausalLM
        ea_model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "EAGLEDrafter":
        cfg = target_model.config
        if ea_model_path is None:
            ea_model_path, source = select_eagle_drafter_for_target(cfg)
        else:
            source = "explicit"
        logger.info(
            "[EAGLE_DRAFTER_SELECT] target_model_type=%r target=%r -> drafter=%r (source=%s)",
            getattr(cfg, "model_type", None),
            getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None),
            ea_model_path,
            source,
        )
        return cls(
            ea_model_path=ea_model_path,
            target_hidden_size=cfg.hidden_size,
            target_vocab_size=cfg.vocab_size,
            target_lm_head=target_model.lm_head,
            target_config=cfg,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _step(
        self,
        hidden_states: torch.Tensor,           # [B, S, H]
        input_ids: torch.LongTensor,           # [B, S]
        position_ids: torch.LongTensor,        # [B, S]
        past_key_values: Optional[DynamicCache],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """One drafter forward; returns ([B, S, H], past_kv)."""
        return self.head(
            hidden_states=hidden_states.to(self.dtype),
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply target LM head to drafter hidden to get vocab logits."""
        return F.linear(
            hidden.to(device=self.lm_head_weight.device, dtype=self.dtype),
            self.lm_head_weight,
            self.lm_head_bias,
        )

    @torch.no_grad()
    def _topk_logprobs(self, logits: torch.Tensor, k: int) -> _TopKLogProbs:
        """Return top-k log-probabilities without materializing full log_softmax."""
        logits_f = logits.float()
        k = min(max(1, int(k)), int(logits_f.shape[-1]))
        top_values, top_indices = torch.topk(logits_f, k=k, dim=-1)
        top_values = top_values - torch.logsumexp(logits_f, dim=-1, keepdim=True)
        return _TopKLogProbs(values=top_values, indices=top_indices)

    @torch.no_grad()
    def _clone_cache(self, cache: Optional[DynamicCache]) -> DynamicCache:
        """Clone the one-layer EAGLE KV cache so candidate branches can diverge."""
        new_cache = DynamicCache(config=self.head_cfg)
        if cache is None or cache.get_seq_length(0) == 0:
            return new_cache
        layer = cache.layers[0]
        new_cache.update(layer.keys.clone(), layer.values.clone(), 0)
        return new_cache

    @torch.no_grad()
    def _merge_prefix_caches_batch(self, caches: Sequence[DynamicCache]) -> DynamicCache:
        """Merge same-length single-row EAGLE prefix caches into one batch cache."""
        if not caches:
            return DynamicCache(config=self.head_cfg)
        if len(caches) == 1:
            return self._clone_cache(caches[0])

        expected_len = int(caches[0].get_seq_length(0))
        ddp_cache_data = []
        for layer_idx in range(len(caches[0].layers)):
            keys: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            for cache in caches:
                if int(cache.get_seq_length(0)) != expected_len:
                    raise ValueError("Cannot batch EAGLE prefix caches with different sequence lengths")
                layer = cache.layers[layer_idx]
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is None or value is None:
                    raise ValueError("Cannot batch an empty EAGLE prefix cache layer")
                keys.append(key)
                values.append(value)
            ddp_cache_data.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        return DynamicCache(ddp_cache_data=ddp_cache_data, config=self.head_cfg)

    @torch.no_grad()
    def _slice_prefix_cache_batch(self, cache: DynamicCache, batch_index: int) -> DynamicCache:
        """Extract one row from a batched one-layer EAGLE cache."""
        ddp_cache_data = []
        for layer_idx in range(len(cache.layers)):
            layer = cache.layers[layer_idx]
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise ValueError("Cannot slice an empty EAGLE prefix cache layer")
            ddp_cache_data.append((
                key[batch_index:batch_index + 1].detach().clone().contiguous(),
                value[batch_index:batch_index + 1].detach().clone().contiguous(),
            ))
        return DynamicCache(ddp_cache_data=ddp_cache_data, config=self.head_cfg)

    @torch.no_grad()
    def _prefill_with_prefix_batch(
        self,
        prefix_hidden_states: torch.Tensor,   # [B, P, H], target hiddens before root
        shifted_input_ids: torch.LongTensor,  # [B, P], tokens 1..root
        *,
        cache_keys: Sequence[int],
    ) -> List[tuple[torch.Tensor, DynamicCache, int]]:
        """Batched variant of _prefill_with_prefix for same-length prefixes.

        The row-wise path is correct but expensive for batch decoding because it
        launches one EAGLE prefix replay per request. Same-length rows can share
        one drafter forward while still keeping independent per-row KV caches.
        """
        assert prefix_hidden_states.ndim == 3
        assert shifted_input_ids.ndim == 2
        assert prefix_hidden_states.shape[:2] == shifted_input_ids.shape
        if shifted_input_ids.shape[1] == 0:
            raise ValueError("EAGLE prefix batch prefill received an empty prefix")
        if len(cache_keys) != int(shifted_input_ids.shape[0]):
            raise ValueError("cache_keys length must match prefix batch size")

        batch_size = int(shifted_input_ids.shape[0])
        total_len = int(shifted_input_ids.shape[1])
        shifted_cpu = shifted_input_ids.detach().cpu()

        states: List[_PrefixCacheState] = []
        valid_states = True
        cache_lens: List[int] = []
        for row, cache_key in enumerate(cache_keys):
            state = self._prefix_states.get(int(cache_key))
            row_valid = (
                state is not None
                and state.cache_ids is not None
                and state.cache_len <= total_len
                and state.cache_ids.shape[0] == 1
                and state.cache_ids.shape[1] >= state.cache_len
                and torch.equal(
                    state.cache_ids[:, :state.cache_len],
                    shifted_cpu[row:row + 1, :state.cache_len],
                )
            )
            if not row_valid:
                valid_states = False
                break
            states.append(state)
            cache_lens.append(int(state.cache_len))

        if not valid_states or len(set(cache_lens)) != 1:
            states = []
            for cache_key in cache_keys:
                state = _PrefixCacheState(
                    cache=DynamicCache(config=self.head_cfg),
                    cache_len=0,
                    cache_ids=torch.empty(1, 0, dtype=torch.long, device="cpu"),
                    last_hidden=None,
                )
                self._prefix_states[int(cache_key)] = state
                states.append(state)
            start = 0
            batch_cache = DynamicCache(config=self.head_cfg)
        else:
            start = cache_lens[0] if cache_lens else 0
            batch_cache = self._merge_prefix_caches_batch([state.cache for state in states])

        if start < total_len:
            new_hidden_states = prefix_hidden_states[:, start:total_len, :]
            new_input_ids = shifted_input_ids[:, start:total_len]
            pos_stack = torch.arange(start, total_len, device=self.device, dtype=torch.long)[None, :]
            pos_stack = pos_stack.expand(batch_size, -1)
            h_drf, batch_cache = self._step(
                hidden_states=new_hidden_states,
                input_ids=new_input_ids,
                position_ids=pos_stack,
                past_key_values=batch_cache,
            )
            last_hiddens = h_drf[:, -1, :].detach()
            for row, (cache_key, state) in enumerate(zip(cache_keys, states)):
                state.cache = self._slice_prefix_cache_batch(batch_cache, row)
                state.cache_len = total_len
                state.cache_ids = shifted_cpu[row:row + 1, :].clone()
                state.last_hidden = last_hiddens[row].detach()
                self._prefix_states[int(cache_key)] = state
        else:
            if any(state.last_hidden is None for state in states):
                raise RuntimeError("EAGLE prefix batch cache is populated without last hidden states")
            last_hiddens = torch.stack(
                [state.last_hidden.to(device=self.device, dtype=self.dtype) for state in states],
                dim=0,
            )

        return [
            (last_hiddens[row], states[row].cache, total_len)
            for row in range(batch_size)
        ]

    @torch.no_grad()
    def _prefill_with_prefix(
        self,
        prefix_hidden_states: torch.Tensor,   # [1, P, H], target hiddens before root
        shifted_input_ids: torch.LongTensor,  # [1, P], tokens 1..root
        *,
        cache_key: int = 0,
    ) -> tuple[torch.Tensor, DynamicCache, int]:
        """Run the official shifted EAGLE prefix once and return root hidden + KV.

        Official EAGLE first runs the drafter over the target hidden states of
        every committed prefix position, paired with the input ids shifted left
        by one token. The last drafter hidden is the predicted hidden state
        for the current root token, and the returned cache is reused by every
        candidate branch for this decode step.
        """
        assert prefix_hidden_states.ndim == 3 and prefix_hidden_states.shape[0] == 1
        assert shifted_input_ids.ndim == 2 and shifted_input_ids.shape[0] == 1
        assert prefix_hidden_states.shape[1] == shifted_input_ids.shape[1]
        if shifted_input_ids.shape[1] == 0:
            raise ValueError("EAGLE prefix prefill received an empty prefix")

        total_len = int(shifted_input_ids.shape[1])
        state = self._prefix_states.get(cache_key)
        if (
            state is None
            or state.cache_ids is None
            or state.cache_len > total_len
            or state.cache_ids.shape[0] != shifted_input_ids.shape[0]
            or state.cache_ids.shape[1] < state.cache_len
            or not torch.equal(
                state.cache_ids[:, :state.cache_len],
                shifted_input_ids[:, :state.cache_len].detach().cpu(),
            )
        ):
            state = _PrefixCacheState(
                cache=DynamicCache(config=self.head_cfg),
                cache_len=0,
                cache_ids=torch.empty(
                    shifted_input_ids.shape[0],
                    0,
                    dtype=torch.long,
                    device="cpu",
                ),
                last_hidden=None,
            )
            self._prefix_states[cache_key] = state

        if state.cache_len < total_len:
            start = state.cache_len
            new_hidden_states = prefix_hidden_states[:, start:total_len, :]
            new_input_ids = shifted_input_ids[:, start:total_len]
            pos_stack = torch.arange(start, total_len, device=self.device, dtype=torch.long)[None, :]
            h_drf, cache = self._step(
                hidden_states=new_hidden_states,
                input_ids=new_input_ids,
                position_ids=pos_stack,
                past_key_values=state.cache,
            )
            state.cache = cache
            state.cache_len = total_len
            state.cache_ids = shifted_input_ids.detach().cpu().clone()
            state.last_hidden = h_drf[0, -1, :].detach()
        elif state.last_hidden is None:
            raise RuntimeError("EAGLE prefix cache is populated without a last hidden state")

        return state.last_hidden, state.cache, total_len

    @torch.no_grad()
    def _advance_cached(
        self,
        prev_hidden: torch.Tensor,            # [H]
        token_id: int,
        position_id: int,
        cache: DynamicCache,
    ) -> tuple[torch.Tensor, DynamicCache]:
        h_drf, cache = self._step(
            hidden_states=prev_hidden[None, None, :],
            input_ids=torch.tensor([[token_id]], device=self.device, dtype=torch.long),
            position_ids=torch.tensor([[position_id]], device=self.device, dtype=torch.long),
            past_key_values=cache,
        )
        return h_drf[0, -1, :], cache

    @torch.no_grad()
    def _hidden_from_cached_path(
        self,
        root_hidden: torch.Tensor,
        root_cache: DynamicCache,
        next_position: int,
        path_tokens: torch.LongTensor,
    ) -> torch.Tensor:
        hidden = root_hidden
        cache = self._clone_cache(root_cache)
        pos = int(next_position)
        for tok in path_tokens.tolist():
            hidden, cache = self._advance_cached(hidden, int(tok), pos, cache)
            pos += 1
        return hidden

    def _tree_attention_mask(
        self,
        tree_mask: torch.Tensor,
        prefix_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an additive EAGLE tree mask for one batched expansion layer."""
        tree_mask = tree_mask.to(device=self.device, dtype=torch.bool)
        neg_inf = torch.finfo(dtype).min
        if tree_mask.ndim == 2:
            q_len, tree_cols = tree_mask.shape
            additive = torch.zeros((q_len, prefix_len + tree_cols), dtype=dtype, device=self.device)
            additive[:, prefix_len:] = additive[:, prefix_len:].masked_fill(~tree_mask, neg_inf)
            return additive.view(1, 1, q_len, prefix_len + tree_cols)
        if tree_mask.ndim == 3:
            batch_size, q_len, tree_cols = tree_mask.shape
            additive = torch.zeros(
                (batch_size, q_len, prefix_len + tree_cols),
                dtype=dtype,
                device=self.device,
            )
            additive[:, :, prefix_len:] = additive[:, :, prefix_len:].masked_fill(~tree_mask, neg_inf)
            return additive.unsqueeze(1)
        raise ValueError(f"EAGLE tree mask must be rank 2 or 3, got shape={tuple(tree_mask.shape)}")

    @torch.no_grad()
    def _build_trees_from_prefix_caches_batched(
        self,
        *,
        jobs: Sequence[_PrefixBuildJob],
        max_candidate_depth: int,
        total_token: int,
        expansion_width: int,
    ) -> List[SpeculativeTree]:
        """Batched EAGLE-2 tree expansion for same-length prefix caches.

        This is the latency-critical path for batch inference. It keeps the
        EAGLE dynamic tree algorithm identical to the single-row path, but runs
        each expansion depth as one `[B, K]` head forward instead of `B`
        independent `[1, K]` forwards.
        """
        if not jobs:
            return []
        if len(jobs) == 1:
            job = jobs[0]
            return [
                self._build_tree_from_prefix_cache(
                    root_tok=job.root_token,
                    root_hidden=job.root_hidden,
                    prefix_cache=job.prefix_cache,
                    prefix_next_pos=job.prefix_next_pos,
                    max_candidate_depth=max_candidate_depth,
                    total_token=total_token,
                    expansion_width=expansion_width,
                )
            ]

        prefix_next_pos = int(jobs[0].prefix_next_pos)
        if any(int(job.prefix_next_pos) != prefix_next_pos for job in jobs):
            raise ValueError("Batched EAGLE tree expansion requires same-length prefix caches")

        batch_size = len(jobs)
        K = max(1, int(expansion_width))
        root_hiddens = torch.stack(
            [job.root_hidden.to(device=self.device, dtype=self.dtype) for job in jobs],
            dim=0,
        )
        root_tokens = [int(job.root_token) for job in jobs]
        work_cache = self._merge_prefix_caches_batch([job.prefix_cache for job in jobs])

        logits0 = self._logits(root_hiddens)
        top0 = self._topk_logprobs(logits0, k=K)
        seed_count = int(top0.indices.shape[-1])

        # PERF (tensorization, codex topology opt B): batch the GPU->CPU transfers.
        # Reading top-k values/indices with one `.tolist()` each is a single CUDA
        # sync; the old per-element `.item()` was O(B*K) separate syncs per depth.
        top0_vals = top0.values.detach().cpu().tolist()      # [B][seed_count]
        top0_idx = top0.indices.detach().cpu().tolist()      # [B][seed_count]
        roots: List[_CandNode] = []
        all_nodes_by_batch: List[List[_CandNode]] = []
        seeds_by_batch: List[List[_CandNode]] = []
        for b in range(batch_size):
            root = _CandNode(
                token_id=root_tokens[b],
                parent=None,
                depth=0,
                log_p=0.0,
                path_log_p=0.0,
                hidden=root_hiddens[b].detach(),
            )
            roots.append(root)
            all_nodes: List[_CandNode] = []
            seeds: List[_CandNode] = []
            for j in range(seed_count):
                lp = float(top0_vals[b][j])
                node = _CandNode(
                    token_id=int(top0_idx[b][j]),
                    parent=root,
                    depth=1,
                    log_p=lp,
                    path_log_p=lp,
                    hidden=None,
                )
                seeds.append(node)
                all_nodes.append(node)
            all_nodes_by_batch.append(all_nodes)
            seeds_by_batch.append(seeds)

        if seed_count == 0:
            return [
                _bind_into_speculative_tree(root_token=root_tokens[b], kept=[])
                for b in range(batch_size)
            ]

        tree_mask = (
            torch.eye(seed_count, dtype=torch.bool, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )

        try:
            for depth in range(1, max_candidate_depth):
                if not seeds_by_batch or not seeds_by_batch[0]:
                    break

                current_seed_count = len(seeds_by_batch[0])
                # PERF (tensorization): build seed_input_ids in one host->device
                # copy instead of per-(b,s) scalar assignment (each of which was a
                # device sync). parent_hiddens are stacked from already-on-device
                # tensors via torch.stack (no per-element sync).
                seed_token_rows: List[List[int]] = []
                parent_hidden_rows: List[torch.Tensor] = []
                for b, seeds in enumerate(seeds_by_batch):
                    if len(seeds) != current_seed_count:
                        raise ValueError("Batched EAGLE expansion requires uniform seed count")
                    seed_token_rows.append([int(seed.token_id) for seed in seeds])
                    ph_row = []
                    for seed in seeds:
                        parent_hidden = (
                            seed.parent.hidden
                            if seed.parent is not None and seed.parent.hidden is not None
                            else roots[b].hidden
                        )
                        ph_row.append(parent_hidden.to(device=self.device, dtype=self.dtype))
                    parent_hidden_rows.append(torch.stack(ph_row, dim=0))
                parent_hiddens = torch.stack(parent_hidden_rows, dim=0)
                seed_input_ids = torch.tensor(
                    seed_token_rows, device=self.device, dtype=torch.long
                )

                position_ids = torch.full(
                    (batch_size, current_seed_count),
                    int(prefix_next_pos + depth - 1),
                    device=self.device,
                    dtype=torch.long,
                )
                attn_mask = self._tree_attention_mask(tree_mask, int(prefix_next_pos), self.dtype)
                out_hidden, work_cache = self._step(
                    hidden_states=parent_hiddens,
                    input_ids=seed_input_ids,
                    position_ids=position_ids,
                    past_key_values=work_cache,
                    attention_mask=attn_mask,
                )
                layer_hidden = out_hidden.detach()
                for b, seeds in enumerate(seeds_by_batch):
                    for s_idx, seed in enumerate(seeds):
                        seed.hidden = layer_hidden[b, s_idx]

                seed_logits = self._logits(layer_hidden.reshape(batch_size * current_seed_count, -1))
                top_k = self._topk_logprobs(seed_logits, k=K)
                top_values = top_k.values.view(batch_size, current_seed_count, -1)
                top_indices = top_k.indices.view(batch_size, current_seed_count, -1)
                child_count = int(top_indices.shape[-1])
                seed_scores = torch.tensor(
                    [
                        [seed.path_log_p for seed in seeds]
                        for seeds in seeds_by_batch
                    ],
                    dtype=top_values.dtype,
                    device=self.device,
                )
                cumulative = top_values + seed_scores[:, :, None]

                # PERF (tensorization): one `.tolist()` per tensor (two CUDA syncs
                # total) instead of B*K*K separate `.item()` calls per depth.
                tv_list = top_values.detach().cpu().tolist()    # [B][seed][child]
                ti_list = top_indices.detach().cpu().tolist()   # [B][seed][child]
                children_by_batch: List[List[List[_CandNode]]] = []
                for b, seeds in enumerate(seeds_by_batch):
                    children_by_seed: List[List[_CandNode]] = []
                    for s_idx, seed in enumerate(seeds):
                        row: List[_CandNode] = []
                        for c in range(child_count):
                            lp = float(tv_list[b][s_idx][c])
                            child = _CandNode(
                                token_id=int(ti_list[b][s_idx][c]),
                                parent=seed,
                                depth=depth + 1,
                                log_p=lp,
                                path_log_p=seed.path_log_p + lp,
                                hidden=None,
                            )
                            row.append(child)
                            all_nodes_by_batch[b].append(child)
                        children_by_seed.append(row)
                    children_by_batch.append(children_by_seed)

                next_count = min(K, cumulative.shape[1] * cumulative.shape[2])
                if next_count <= 0:
                    break
                top_next = torch.topk(cumulative.reshape(batch_size, -1), k=next_count, dim=-1)
                parent_indices = torch.div(top_next.indices, child_count, rounding_mode="floor")
                child_indices = top_next.indices.remainder(child_count)

                # PERF (tensorization): one `.tolist()` for the whole [B, next_count]
                # selection instead of 2*B per-row `.tolist()` calls.
                parent_idx_list = parent_indices.detach().cpu().tolist()
                child_idx_list = child_indices.detach().cpu().tolist()
                next_seeds_by_batch: List[List[_CandNode]] = []
                for b in range(batch_size):
                    next_seeds: List[_CandNode] = []
                    for parent_idx, child_idx in zip(parent_idx_list[b], child_idx_list[b]):
                        next_seeds.append(children_by_batch[b][int(parent_idx)][int(child_idx)])
                    next_seeds_by_batch.append(next_seeds)

                row_ids = torch.arange(batch_size, device=self.device)[:, None]
                selected_parent_mask = tree_mask[row_ids, parent_indices.to(device=self.device)]
                child_eye = (
                    torch.eye(next_count, dtype=torch.bool, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
                tree_mask = torch.cat((selected_parent_mask, child_eye), dim=2)
                seeds_by_batch = next_seeds_by_batch
        finally:
            if hasattr(work_cache, "crop"):
                work_cache.crop(int(prefix_next_pos))

        m = max(0, int(total_token) - 1)
        trees: List[SpeculativeTree] = []
        for b in range(batch_size):
            kept = _topm_global(all_nodes_by_batch[b], m=m)
            kept = _close_under_parents(kept)
            trees.append(_bind_into_speculative_tree(root_token=root_tokens[b], kept=kept))
        return trees

    @torch.no_grad()
    def _build_tree_from_prefix_cache(
        self,
        *,
        root_tok: int,
        root_hidden: torch.Tensor,
        prefix_cache: DynamicCache,
        prefix_next_pos: int,
        max_candidate_depth: int,
        total_token: int,
        expansion_width: int,
    ) -> SpeculativeTree:
        """EAGLE-2 tree expansion with the official layer-wise tree mask.

        This avoids replaying every root-to-node path independently. Each
        iteration advances the selected top-k frontier nodes in one drafter
        forward, while the tree mask prevents sibling nodes from attending to
        each other.
        """
        K = max(1, int(expansion_width))
        root_drafter_hidden = root_hidden
        logits0 = self._logits(root_drafter_hidden[None, :])
        top0 = self._topk_logprobs(logits0, k=K)

        root = _CandNode(
            token_id=root_tok,
            parent=None,
            depth=0,
            log_p=0.0,
            path_log_p=0.0,
            hidden=root_drafter_hidden,
        )
        all_nodes: List[_CandNode] = []
        seeds: List[_CandNode] = []
        for j in range(top0.indices.shape[-1]):
            tok = int(top0.indices[0, j].item())
            lp = float(top0.values[0, j].item())
            node = _CandNode(
                token_id=tok,
                parent=root,
                depth=1,
                log_p=lp,
                path_log_p=lp,
                hidden=None,
            )
            seeds.append(node)
            all_nodes.append(node)

        can_crop_prefix_cache = hasattr(prefix_cache, "crop")
        work_cache = prefix_cache if can_crop_prefix_cache else self._clone_cache(prefix_cache)
        tree_mask = torch.eye(len(seeds), dtype=torch.bool, device=self.device)

        try:
            for depth in range(1, max_candidate_depth):
                if not seeds:
                    break

                parent_hiddens = torch.stack(
                    [
                        (s.parent.hidden if s.parent is not None and s.parent.hidden is not None else root_drafter_hidden)
                        for s in seeds
                    ],
                    dim=0,
                ).to(device=self.device, dtype=self.dtype)
                seed_input_ids = torch.tensor(
                    [[s.token_id for s in seeds]],
                    device=self.device,
                    dtype=torch.long,
                )
                position_ids = torch.full(
                    (1, len(seeds)),
                    int(prefix_next_pos + depth - 1),
                    device=self.device,
                    dtype=torch.long,
                )
                attn_mask = self._tree_attention_mask(tree_mask, int(prefix_next_pos), self.dtype)
                out_hidden, work_cache = self._step(
                    hidden_states=parent_hiddens[None, :, :],
                    input_ids=seed_input_ids,
                    position_ids=position_ids,
                    past_key_values=work_cache,
                    attention_mask=attn_mask,
                )
                layer_hidden = out_hidden[0]
                for idx, seed in enumerate(seeds):
                    seed.hidden = layer_hidden[idx].detach()

                seed_logits = self._logits(layer_hidden)
                top_k = self._topk_logprobs(seed_logits, k=K)
                seed_scores = torch.tensor(
                    [s.path_log_p for s in seeds],
                    dtype=top_k.values.dtype,
                    device=self.device,
                )
                cumulative = top_k.values + seed_scores[:, None]

                children_by_seed: List[List[_CandNode]] = []
                for seed_idx, seed in enumerate(seeds):
                    row: List[_CandNode] = []
                    for c in range(K):
                        tok = int(top_k.indices[seed_idx, c].item())
                        lp = float(top_k.values[seed_idx, c].item())
                        child = _CandNode(
                            token_id=tok,
                            parent=seed,
                            depth=depth + 1,
                            log_p=lp,
                            path_log_p=seed.path_log_p + lp,
                            hidden=None,
                        )
                        row.append(child)
                        all_nodes.append(child)
                    children_by_seed.append(row)

                next_count = min(K, cumulative.numel())
                if next_count <= 0:
                    break
                top_next = torch.topk(cumulative.reshape(-1), k=next_count, dim=-1)
                parent_indices = torch.div(top_next.indices, K, rounding_mode="floor")
                child_indices = top_next.indices.remainder(K)

                next_seeds: List[_CandNode] = []
                for parent_idx, child_idx in zip(parent_indices.tolist(), child_indices.tolist()):
                    next_seeds.append(children_by_seed[int(parent_idx)][int(child_idx)])
                tree_mask = torch.cat(
                    (
                        tree_mask[parent_indices.to(device=self.device)],
                        torch.eye(next_count, dtype=torch.bool, device=self.device),
                    ),
                    dim=1,
                )
                seeds = next_seeds
        finally:
            if can_crop_prefix_cache:
                work_cache.crop(int(prefix_next_pos))

        m = max(0, int(total_token) - 1)
        kept = _topm_global(all_nodes, m=m)
        kept = _close_under_parents(kept)
        return _bind_into_speculative_tree(root_token=root_tok, kept=kept)

    @torch.no_grad()
    def _replay_path(
        self,
        path_tokens: torch.LongTensor,        # [P]   ids along path (excluding the root token)
        target_hidden_root: torch.Tensor,     # [H]   target last-layer hidden of the path's root token
        root_token_id: int,
        base_pos: int,                        # absolute position of root token
    ) -> torch.Tensor:
        """Re-run the EAGLE head autoregressively along the path, threading
        the drafter's own hidden output as the next position's hidden input.

        At each step:
          h_drafter_t = layer( fc(concat(emb(y_t), h_input_t)), past=KV_{<t}, pos=base_pos+t )
        where:
          h_input_0 = target_hidden_root  (the only target-aligned point)
          h_input_t = h_drafter_{t-1}     (drafter's own previous output)

        This matches EAGLE's official ``topK_genrate`` exactly: see
        ``eagle.model.cnets.Model.topK_genrate``, which iterates with
        ``hidden_states = out_hidden`` carried across depths and a
        ``past_key_values`` cache for the head's self-attention.

        Returns: ``[P+1, H]`` drafter hidden states at each path position.
        """
        seq_ids = torch.cat(
            [torch.tensor([root_token_id], device=self.device, dtype=torch.long), path_tokens],
            dim=0,
        )                                                              # [P+1]

        out_hiddens = []
        cache = DynamicCache(config=self.head_cfg)
        hidden = target_hidden_root
        for t, tok in enumerate(seq_ids.tolist()):
            hidden, cache = self._advance_cached(hidden, int(tok), base_pos + t, cache)
            out_hiddens.append(hidden)

        return torch.stack(out_hiddens, dim=0)                         # [P+1, H]

    @torch.no_grad()
    def build_trees_parallel(
        self,
        input_ids: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        beam_width: Union[int, Sequence[int]] = 4,
        max_depth: int = 6,
        *,
        prev_last_hidden: Optional[torch.Tensor] = None,
        prev_last_token: Optional[torch.LongTensor] = None,
        prefix_hidden_states: Optional[torch.Tensor] = None,
        total_token: Optional[int] = None,
        topk_per_step: Optional[int] = None,
        do_sample: bool = False,
        # EAGLE-2 paper uses a "global top-m" ≈ total_token; we expose it.
        # The other budget-pruner kwargs are accepted for signature parity but ignored.
        tree_budget: Optional[int] = None,
        tree_min_log_prob: Optional[float] = None,
        **_,
    ) -> List[SpeculativeTree]:
        """EAGLE-2 dynamic-tree drafting.

        Implementation faithfulness:
          - When target prefix hidden states are available, we run the
            official shifted EAGLE prefix once into a KV cache, reuse that
            cache for candidate branches, then crop it back to the prefix.
            Without prefix hidden states
            we fall back to advancing from the root with a fresh KV cache.
            Both paths preserve EAGLE's trained conditioning:
            auto-regressive on its own previous hidden, target hidden at
            the aligned root/prefix positions, RoPE positions = base + t.
          - This is O(K · D) per expansion in head-FLOPs (K seeds × D depth);
            since the head is a single Llama layer, it's still cheap relative
            to the target verify forward.

        Args:
          ``prev_last_hidden`` [B, H]: target last-layer hidden of the last
            committed token (the path root). EAGLE conditioning input.
          ``prev_last_token`` [B]: token id at the path root.
          ``total_token`` int: global tree-size budget including root. When
            unset, BloomBee's requested draft-token budget is inferred from
            ``beam_width`` and ``max_depth`` so ``depth=5,width=1`` verifies
            five draft nodes, not EAGLE's paper-sized 60-node tree.
          ``topk_per_step`` int: per-layer top-k expansion count.
        """
        if (
            prefix_hidden_states is None
            and (prev_last_hidden is None or prev_last_token is None)
        ):
            # First iteration before we have a target hidden state — fall back to
            # an empty tree; caller will detect zero tree_tokens and use the
            # AR fallback path. This keeps the very first decode step safe.
            B = int(input_ids.shape[0])
            out: List[SpeculativeTree] = []
            for b in range(B):
                root = int(input_ids[b, max(0, int(seq_lengths[b].item()) - 1)].item())
                out.append(SpeculativeTree(root, request_id=f"eagle2_warmup_{b}"))
            return out

        if prev_last_hidden is not None:
            prev_last_hidden = prev_last_hidden.to(self.device).to(self.dtype)
        if prev_last_token is not None:
            prev_last_token = prev_last_token.to(self.device).long()
        if prefix_hidden_states is not None:
            prefix_hidden_states = prefix_hidden_states.to(self.device).to(self.dtype)

        explicit_tree_budget = tree_budget is not None or total_token is not None
        draft_budget = (
            max(0, int(tree_budget))
            if tree_budget is not None
            else _default_draft_token_budget(beam_width, max_depth)
        )
        if total_token is None:
            total_token = draft_budget + 1
        else:
            total_token = max(1, int(total_token))

        max_candidate_depth = _eagle2_max_candidate_depth(
            max_depth,
            explicit_tree_budget=explicit_tree_budget,
        )
        if explicit_tree_budget:
            expansion_width = (
                int(topk_per_step)
                if topk_per_step is not None
                else _default_eagle_topk_per_step(
                    int(total_token),
                    explicit_tree_budget=explicit_tree_budget,
                    do_sample=bool(do_sample),
                )
            )
        elif isinstance(beam_width, (list, tuple)):
            expansion_width = max((int(w) for w in beam_width[:max_depth]), default=1)
        else:
            expansion_width = int(beam_width)
        # ``topk_per_step`` controls how many candidates the EAGLE-2 paper path
        # expands before global rerank. Plain BloomBee calls should still honor
        # their requested tree shape: depth=5,width=1 means one 5-token path,
        # not five shallow siblings picked by the reranker.
        K_child = max(1, min(expansion_width, max(1, total_token - 1)))

        batch_size = int(input_ids.shape[0])
        for cache_key in list(self._prefix_states.keys()):
            if cache_key >= batch_size:
                del self._prefix_states[cache_key]

        results: List[Optional[SpeculativeTree]] = [None] * batch_size
        prefix_jobs: List[_PrefixBuildJob] = []
        batched_prefix_rows: set[int] = set()
        if prefix_hidden_states is not None and batch_size > 1:
            prefix_lengths = [max(0, int(seq_lengths[b].item()) - 1) for b in range(batch_size)]
            common_prefix_len = prefix_lengths[0] if prefix_lengths else 0
            can_batch_prefix = (
                common_prefix_len > 0
                and all(length == common_prefix_len for length in prefix_lengths)
                and prefix_hidden_states.shape[1] >= common_prefix_len
            )
            if can_batch_prefix:
                try:
                    shifted_ids_batch = input_ids[:, 1:common_prefix_len + 1].to(self.device)
                    prefix_prefills = self._prefill_with_prefix_batch(
                        prefix_hidden_states[:, :common_prefix_len, :],
                        shifted_ids_batch,
                        cache_keys=list(range(batch_size)),
                    )
                    for b, (prefix_root_hidden, prefix_cache, prefix_next_pos) in enumerate(prefix_prefills):
                        root_pos = int(seq_lengths[b].item()) - 1
                        root_tok = (
                            int(prev_last_token[b].item())
                            if prev_last_token is not None
                            else int(input_ids[b, root_pos].item())
                        )
                        prefix_jobs.append(
                            _PrefixBuildJob(
                                batch_index=b,
                                root_token=root_tok,
                                root_hidden=prefix_root_hidden,
                                prefix_cache=prefix_cache,
                                prefix_next_pos=prefix_next_pos,
                            )
                        )
                        batched_prefix_rows.add(b)
                except Exception as e:
                    logger.debug(
                        "[EAGLEDrafter] batched prefix prefill failed; falling back row-wise: %s",
                        e,
                        exc_info=True,
                    )

        for b in range(batch_size):
            if b in batched_prefix_rows:
                continue
            root_pos = int(seq_lengths[b].item()) - 1
            base_pos = max(0, root_pos - 1)
            if prev_last_token is not None:
                root_tok = int(prev_last_token[b].item())
            else:
                root_tok = int(input_ids[b, root_pos].item())

            prefix_len = root_pos
            use_prefix = (
                prefix_hidden_states is not None
                and prefix_len > 0
                and prefix_hidden_states.shape[1] >= prefix_len
            )
            prefix_root_hidden: Optional[torch.Tensor] = None
            prefix_cache: Optional[DynamicCache] = None
            prefix_next_pos = 0
            if use_prefix:
                shifted_ids = input_ids[b:b + 1, 1:prefix_len + 1].to(self.device)
                prefix_root_hidden, prefix_cache, prefix_next_pos = self._prefill_with_prefix(
                    prefix_hidden_states[b:b + 1, :prefix_len, :],
                    shifted_ids,
                    cache_key=b,
                )
                prefix_jobs.append(
                    _PrefixBuildJob(
                        batch_index=b,
                        root_token=root_tok,
                        root_hidden=prefix_root_hidden,
                        prefix_cache=prefix_cache,
                        prefix_next_pos=prefix_next_pos,
                    )
                )
                continue

            def hidden_for_path(path_tokens: torch.LongTensor) -> torch.Tensor:
                if prev_last_hidden is None:
                    raise ValueError("EAGLE replay requires prev_last_hidden without prefix_hidden_states")
                path_hidden = self._replay_path(
                    path_tokens=path_tokens,
                    target_hidden_root=prev_last_hidden[b],
                    root_token_id=root_tok,
                    base_pos=base_pos,
                )
                return path_hidden[-1]

            # ---- depth 0: predict depth-1 candidates from root ----
            root_drafter_hidden = hidden_for_path(
                torch.empty(0, device=self.device, dtype=torch.long)
            )
            logits0 = self._logits(root_drafter_hidden[None, :])    # [1, V]
            top0 = self._topk_logprobs(logits0, k=K_child)

            root = _CandNode(
                token_id=root_tok,
                parent=None,
                depth=0,
                log_p=0.0,
                path_log_p=0.0,
                hidden=root_drafter_hidden,
            )
            all_nodes: List[_CandNode] = []
            layer_nodes: List[List[_CandNode]] = [[root]]
            children0: List[_CandNode] = []
            for j in range(top0.indices.shape[-1]):
                tok = int(top0.indices[0, j].item())
                lp = float(top0.values[0, j].item())
                children0.append(_CandNode(
                    token_id=tok, parent=root, depth=1,
                    log_p=lp, path_log_p=lp,
                    hidden=None,
                ))
            layer_nodes.append(children0)
            all_nodes.extend(children0)

            # ---- depth 1..max_depth-1: expand top-k seeds from the latest layer ----
            for depth in range(1, max_candidate_depth):
                latest = layer_nodes[depth]
                if not latest:
                    break
                seeds = _topk_per_layer_indices(latest, k=K_child)

                # For each seed, advance the branch through the head with a
                # KV cache. The seed's hidden output is used to compute its
                # top-K_child next-token candidates.
                seed_topk_values: List[torch.Tensor] = []
                seed_topk_indices: List[torch.Tensor] = []
                for s in seeds:
                    # Walk parents to get path tokens (root excluded)
                    chain: List[int] = []
                    cur = s
                    while cur is not None and cur.parent is not None:
                        chain.append(cur.token_id)
                        cur = cur.parent
                    chain.reverse()
                    path_tokens = torch.tensor(chain, device=self.device, dtype=torch.long)
                    s.hidden = hidden_for_path(path_tokens)
                    seed_logits = self._logits(s.hidden[None, :])    # [1, V]
                    seed_topk = self._topk_logprobs(seed_logits, k=K_child)
                    seed_topk_values.append(seed_topk.values[0])
                    seed_topk_indices.append(seed_topk.indices[0])

                # Stack and topK_child per seed
                top_k = _TopKLogProbs(
                    values=torch.stack(seed_topk_values, dim=0),
                    indices=torch.stack(seed_topk_indices, dim=0),
                )

                new_layer: List[_CandNode] = []
                for j, s in enumerate(seeds):
                    for c in range(K_child):
                        tok = int(top_k.indices[j, c].item())
                        lp = float(top_k.values[j, c].item())
                        new_layer.append(_CandNode(
                            token_id=tok, parent=s, depth=depth + 1,
                            log_p=lp, path_log_p=s.path_log_p + lp,
                            hidden=None,
                        ))
                layer_nodes.append(new_layer)
                all_nodes.extend(new_layer)

            # Global top-m rerank + parent closure
            m = max(0, total_token - 1)  # exclude root from the budget
            kept = _topm_global(all_nodes, m=m)
            kept = _close_under_parents(kept)

            tree = _bind_into_speculative_tree(
                root_token=root_tok,
                kept=kept,
            )
            results[b] = tree

        if prefix_jobs:
            jobs_by_prefix_len: Dict[int, List[_PrefixBuildJob]] = {}
            for job in prefix_jobs:
                jobs_by_prefix_len.setdefault(int(job.prefix_next_pos), []).append(job)

            for jobs in jobs_by_prefix_len.values():
                try:
                    built_trees = self._build_trees_from_prefix_caches_batched(
                        jobs=jobs,
                        max_candidate_depth=max_candidate_depth,
                        total_token=total_token,
                        expansion_width=K_child,
                    )
                except Exception as e:
                    logger.debug(
                        "[EAGLEDrafter] batched prefix expansion failed; falling back row-wise: %s",
                        e,
                        exc_info=True,
                    )
                    built_trees = [
                        self._build_tree_from_prefix_cache(
                            root_tok=job.root_token,
                            root_hidden=job.root_hidden,
                            prefix_cache=job.prefix_cache,
                            prefix_next_pos=job.prefix_next_pos,
                            max_candidate_depth=max_candidate_depth,
                            total_token=total_token,
                            expansion_width=K_child,
                        )
                        for job in jobs
                    ]

                for job, tree in zip(jobs, built_trees):
                    results[job.batch_index] = tree

        if any(tree is None for tree in results):
            missing = [idx for idx, tree in enumerate(results) if tree is None]
            raise RuntimeError(f"EAGLE failed to build speculative trees for batch rows {missing}")

        return [tree for tree in results if tree is not None]
