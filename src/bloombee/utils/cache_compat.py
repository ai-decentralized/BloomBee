"""Compatibility helpers for transformers 4.x / 5.x Cache API differences.

tf 4.x DynamicCache: stores KV in .key_cache/.value_cache lists, tracks ._seen_tokens
tf 5.x DynamicCache: stores KV in .layers[] (DynamicLayer objects with .keys/.values),
    no .key_cache/.value_cache/._ seen_tokens attributes.
"""

import inspect
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache

# ── Feature detection (NOT version-based — tf 4.57+ backported the new API) ───
# Probe a real DynamicCache instance to decide which API to use.
_probe_cache = DynamicCache()
_HAS_LAYERS_API = hasattr(_probe_cache, "layers")
_HAS_KEY_CACHE = hasattr(_probe_cache, "key_cache")
del _probe_cache

# Public flag: True when DynamicCache uses the new .layers[] API
# (covers tf 5.x AND tf 4.57+ which backported it)
_IS_TF5 = _HAS_LAYERS_API

# Legacy detection (kept for backward compat with other callers)
_CACHE_INIT_PARAMS = inspect.signature(Cache.__init__).parameters
_CACHE_USES_LAYER_API = "layers" in _CACHE_INIT_PARAMS


def init_cache_base(cache: Cache) -> None:
    """Call Cache.__init__ in a way that works on both tf 4.x and 5.x."""
    if _CACHE_USES_LAYER_API:
        Cache.__init__(cache, layers=[])
    else:
        Cache.__init__(cache)


def make_dynamic_cache() -> DynamicCache:
    """Create an empty DynamicCache compatible with both tf 4.x and 5.x."""
    return DynamicCache()


# ── Per-block cache helpers (used by model block wrappers) ─────────────────────

def make_past_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    seen_tokens: Optional[int] = None,
) -> DynamicCache:
    """Create a DynamicCache pre-populated with past KV at *layer_idx*.

    Works on both tf 4.x and 5.x.

    Args:
        key_states:  [B, H, S, D] past key tensor
        value_states: [B, H, S, D] past value tensor
        layer_idx:   which transformer layer this cache belongs to
        seen_tokens: total tokens seen so far (used by tf 4.x only)
    """
    dc = DynamicCache()

    if _IS_TF5:
        # tf 5.x: use update() which auto-creates DynamicLayer objects
        # Layers 0..layer_idx-1 get empty (uninitialized) DynamicLayers
        # Layer layer_idx gets populated with the past KV
        dc.update(key_states, value_states, layer_idx=layer_idx)
    else:
        # tf 4.x: populate key_cache/value_cache lists directly
        dc.key_cache = [torch.empty(0) for _ in range(layer_idx)] + [key_states]
        dc.value_cache = [torch.empty(0) for _ in range(layer_idx)] + [value_states]
        if seen_tokens is not None:
            dc._seen_tokens = seen_tokens

    return dc


def make_empty_kv_cache(layer_idx: int = 0) -> DynamicCache:
    """Create an empty DynamicCache for the initial use_cache=True pass.

    In both tf 4.x and 5.x, DynamicCache() + update() auto-extends,
    so no pre-population is strictly needed.  However tf 4.x code in some
    HF model implementations checks ``len(key_cache) <= layer_idx``, so we
    keep the old pre-population pattern there for safety.
    """
    dc = DynamicCache()
    if not _IS_TF5:
        # tf 4.x: pre-fill with None so key_cache[layer_idx] exists after update()
        dc.key_cache = [None] * layer_idx
        dc.value_cache = [None] * layer_idx
    return dc


def read_kv_from_cache(
    cache: DynamicCache,
    layer_idx: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Read the full (past + new) KV tensors from *cache* at *layer_idx*.

    Returns ``(key, value)`` each of shape ``[B, H, S_total, D]``,
    or ``(None, None)`` if the cache has no data at that index.
    """
    if _IS_TF5:
        # tf 5.x: KV lives in layers[].keys / layers[].values
        if layer_idx < len(cache.layers):
            layer = cache.layers[layer_idx]
            if getattr(layer, "is_initialized", False) and layer.keys.numel() > 0:
                return (layer.keys, layer.values)
        # Fallback: scan backwards for any populated layer
        for i in range(len(cache.layers) - 1, -1, -1):
            layer = cache.layers[i]
            if getattr(layer, "is_initialized", False) and layer.keys.numel() > 0:
                return (layer.keys, layer.values)
        return (None, None)
    else:
        # tf 4.x: KV lives in key_cache[] / value_cache[]
        if hasattr(cache, "key_cache") and layer_idx < len(cache.key_cache):
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
            if k is not None and k.dim() == 4:
                return (k, v)
        # Fallback: scan backwards
        if hasattr(cache, "key_cache"):
            for i in range(len(cache.key_cache) - 1, -1, -1):
                t = cache.key_cache[i]
                if t is not None and t.dim() == 4:
                    return (t, cache.value_cache[i])
        return (None, None)
