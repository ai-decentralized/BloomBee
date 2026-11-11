"""Sequence-axis concatenation for GLM compressed MLA cache chunks.

GLM-4.7-Flash stores two latents with different widths:
kv_nope is [B, 1, S, kv_lora_rank] (512) and k_rot is [B, 1, S, qk_rope_head_dim] (64).
The ordinary KV chunk path views both tensors with one shared head dim, which
cannot represent this pair.
"""

from typing import List, Optional, Sequence, Tuple

import torch

MLAPair = Tuple[torch.Tensor, torch.Tensor]


def _check_mla_pair(kv_nope: torch.Tensor, k_rot: torch.Tensor) -> None:
    for name, tensor in (("kv_nope", kv_nope), ("k_rot", k_rot)):
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
            raise RuntimeError(
                f"GLM MLA {name} must be rank-4 [batch, heads, seq, dim], got {type(tensor).__name__} shape {shape}"
            )
    if kv_nope.shape[:3] != k_rot.shape[:3]:
        raise RuntimeError(
            "GLM MLA kv_nope and k_rot disagree on batch, heads, or sequence: "
            f"{tuple(kv_nope.shape)} vs {tuple(k_rot.shape)}"
        )


def concat_mla_states(
    left: Optional[MLAPair],
    right: Optional[MLAPair],
) -> Optional[MLAPair]:
    """Concatenate two MLA pairs on the sequence axis. Either side may be absent."""
    if right is None:
        return left
    if left is None:
        _check_mla_pair(*right)
        return right

    left_kv, left_rot = left
    right_kv, right_rot = right
    _check_mla_pair(left_kv, left_rot)
    _check_mla_pair(right_kv, right_rot)
    if left_kv.shape[0] != right_kv.shape[0] or left_kv.shape[1] != right_kv.shape[1]:
        raise RuntimeError(
            "GLM MLA chunk batch/head layout does not match the running cache: "
            f"past={tuple(left_kv.shape)} new={tuple(right_kv.shape)}"
        )
    if left_kv.shape[-1] != right_kv.shape[-1] or left_rot.shape[-1] != right_rot.shape[-1]:
        raise RuntimeError(
            "GLM MLA chunk latent widths do not match the running cache: "
            f"past=({left_kv.shape[-1]}, {left_rot.shape[-1]}) "
            f"new=({right_kv.shape[-1]}, {right_rot.shape[-1]})"
        )
    return (
        torch.cat((left_kv, right_kv), dim=2),
        torch.cat((left_rot, right_rot), dim=2),
    )


def concat_mla_chunks(chunks: Sequence[MLAPair]) -> Optional[MLAPair]:
    """Concatenate only the newly generated chunks, excluding any prior prefix."""
    if not chunks:
        return None
    # Chunks were validated when recorded. Concatenate each latent once rather
    # than repeatedly copying a growing prefix on long requests.
    kv_nope, k_rot = zip(*chunks)
    return torch.cat(kv_nope, dim=2), torch.cat(k_rot, dim=2)


def record_glm_mla_chunk(
    layer_past: Optional[MLAPair],
    new_kvs: Optional[MLAPair],
    chunks: List[MLAPair],
) -> Optional[MLAPair]:
    """Append this chunk's new latents and return the cache the next chunk must attend to.

    ``chunks`` collects only the tokens produced by this request. ``layer_past``
    may already hold a prefix loaded from the persistent cache; that prefix is
    carried forward for the next forward, but it is not written again.
    """
    if new_kvs is None:
        raise RuntimeError("GLM MLA chunked forward returned no cache")
    kv_nope, k_rot = new_kvs
    _check_mla_pair(kv_nope, k_rot)
    # The block returns views into a temporary DynamicCache. Clone before the
    # next chunk drops that cache.
    chunks.append((kv_nope.detach().clone(), k_rot.detach().clone()))
    return concat_mla_states(layer_past, (kv_nope, k_rot))
