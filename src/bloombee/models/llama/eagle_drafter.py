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
_EAGLE_DEPTH_ENV = "BLOOMBEE_EAGLE_DEPTH"

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
    kept_sorted = sorted(kept, key=lambda n: (n.depth, id(n)))
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
        # The distributed BloomBee client often keeps lm_head on CPU. Clone a
        # read-only projection onto the drafter device instead of moving the
        # target model's module and accidentally perturbing normal generation.
        self.lm_head_weight = target_lm_head.weight.detach().to(
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        bias = getattr(target_lm_head, "bias", None)
        self.lm_head_bias = (
            bias.detach().to(device=self.device, dtype=self.dtype).contiguous()
            if torch.is_tensor(bias)
            else None
        )
        self.uses_eagle_hidden_states = True
        # AutoDistributedSpeculativeModel.generate uses this when callers do
        # not pass tree_budget. The EAGLE-2 paper default is total-token=60
        # including the root, while BloomBee's tree_budget counts draft nodes.
        self.default_tree_budget = max(1, int(os.environ.get(_EAGLE_TREE_BUDGET_ENV, "59")))
        self._prefix_cache: Optional[DynamicCache] = None
        self._prefix_cache_len = 0
        self._prefix_cache_ids: Optional[torch.LongTensor] = None
        self._prefix_last_hidden: Optional[torch.Tensor] = None

        self._load_eagle_weights(ea_model_path)

    def _load_eagle_weights(self, path: str) -> None:
        """Load yuhuili-style EAGLE weights into ``self.head``.

        The official EAGLE checkpoint stores `embed_tokens.weight`,
        `fc.{weight,bias}`, and `layers.0.*` (one decoder layer). We map those
        onto our `EAGLEHead` (which has `embed_tokens`, `fc`, `layer`) and load
        as a missing-key-tolerant state_dict.
        """
        from huggingface_hub import snapshot_download

        repo_path = snapshot_download(
            repo_id=path,
            allow_patterns=["*.bin", "*.safetensors", "config.json"],
        )
        import os
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
        return F.linear(hidden.to(self.dtype), self.lm_head_weight, self.lm_head_bias)

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
    def _prefill_with_prefix(
        self,
        prefix_hidden_states: torch.Tensor,   # [1, P, H], target hiddens before root
        shifted_input_ids: torch.LongTensor,  # [1, P], tokens 1..root
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
        if (
            self._prefix_cache is None
            or self._prefix_cache_ids is None
            or self._prefix_cache_len > total_len
            or self._prefix_cache_ids.shape[0] != shifted_input_ids.shape[0]
            or self._prefix_cache_ids.shape[1] < self._prefix_cache_len
            or not torch.equal(
                self._prefix_cache_ids[:, :self._prefix_cache_len],
                shifted_input_ids[:, :self._prefix_cache_len].detach().cpu(),
            )
        ):
            self._prefix_cache = DynamicCache(config=self.head_cfg)
            self._prefix_cache_len = 0
            self._prefix_cache_ids = torch.empty(
                shifted_input_ids.shape[0],
                0,
                dtype=torch.long,
                device="cpu",
            )
            self._prefix_last_hidden = None

        if self._prefix_cache_len < total_len:
            start = self._prefix_cache_len
            new_hidden_states = prefix_hidden_states[:, start:total_len, :]
            new_input_ids = shifted_input_ids[:, start:total_len]
            pos_stack = torch.arange(start, total_len, device=self.device, dtype=torch.long)[None, :]
            h_drf, cache = self._step(
                hidden_states=new_hidden_states,
                input_ids=new_input_ids,
                position_ids=pos_stack,
                past_key_values=self._prefix_cache,
            )
            self._prefix_cache = cache
            self._prefix_cache_len = total_len
            self._prefix_cache_ids = shifted_input_ids.detach().cpu().clone()
            self._prefix_last_hidden = h_drf[0, -1, :].detach()
        elif self._prefix_last_hidden is None:
            raise RuntimeError("EAGLE prefix cache is populated without a last hidden state")

        return self._prefix_last_hidden, self._prefix_cache, total_len

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
        q_len, tree_cols = tree_mask.shape
        neg_inf = torch.finfo(dtype).min
        additive = torch.zeros((q_len, prefix_len + tree_cols), dtype=dtype, device=self.device)
        additive[:, prefix_len:] = additive[:, prefix_len:].masked_fill(~tree_mask, neg_inf)
        return additive.view(1, 1, q_len, prefix_len + tree_cols)

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
        logp0 = F.log_softmax(logits0.float(), dim=-1)
        top0 = torch.topk(logp0, k=K, dim=-1)

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

        work_cache = self._clone_cache(prefix_cache)
        tree_mask = torch.eye(len(seeds), dtype=torch.bool, device=self.device)

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
            seed_logp = F.log_softmax(seed_logits.float(), dim=-1)
            top_k = torch.topk(seed_logp, k=K, dim=-1)
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
        topk_per_step: int = 10,
        # EAGLE-2 paper uses a "global top-m" ≈ total_token; we expose it.
        # The other budget-pruner kwargs are accepted for signature parity but ignored.
        tree_budget: Optional[int] = None,
        tree_min_log_prob: Optional[float] = None,
        **_,
    ) -> List[SpeculativeTree]:
        """EAGLE-2 dynamic-tree drafting (per-batch, sequential).

        Implementation faithfulness:
          - When target prefix hidden states are available, we run the
            official shifted EAGLE prefix once into a KV cache, then clone
            that cache for candidate branches. Without prefix hidden states
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
            expansion_width = int(topk_per_step)
        elif isinstance(beam_width, (list, tuple)):
            expansion_width = max((int(w) for w in beam_width[:max_depth]), default=1)
        else:
            expansion_width = int(beam_width)
        # ``topk_per_step`` controls how many candidates the EAGLE-2 paper path
        # expands before global rerank. Plain BloomBee calls should still honor
        # their requested tree shape: depth=5,width=1 means one 5-token path,
        # not five shallow siblings picked by the reranker.
        K_child = max(1, min(expansion_width, max(1, total_token - 1)))

        results: List[SpeculativeTree] = []
        for b in range(int(input_ids.shape[0])):
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
                )
                results.append(
                    self._build_tree_from_prefix_cache(
                        root_tok=root_tok,
                        root_hidden=prefix_root_hidden,
                        prefix_cache=prefix_cache,
                        prefix_next_pos=prefix_next_pos,
                        max_candidate_depth=max_candidate_depth,
                        total_token=total_token,
                        expansion_width=K_child,
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
            logp0 = F.log_softmax(logits0.float(), dim=-1)
            top0 = torch.topk(logp0, k=K_child, dim=-1)

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
                seed_logps: List[torch.Tensor] = []
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
                    seed_logps.append(F.log_softmax(seed_logits.float(), dim=-1)[0])

                # Stack and topK_child per seed
                seed_logp = torch.stack(seed_logps, dim=0)            # [K, V]
                top_k = torch.topk(seed_logp, k=K_child, dim=-1)

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
            results.append(tree)

        return results
