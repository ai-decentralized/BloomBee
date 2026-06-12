"""Micro-batch tensor operations: range computation, batch splitting and
output merging. Split out of microbatch_config (which keeps only switches and
logging helpers)."""

import logging
from typing import Tuple, List, Optional, Any

import torch

from bloombee.utils.microbatch_config import (
    MBPIPE_LOG_PREFIX,
    get_micro_batch_size,
    is_microbatch_enabled,
    mbpipe_info_logs_enabled,
)

# =============================================================================
# Micro-batch Splitting and Merging Utilities
# =============================================================================

def should_split_batch(batch_size: int) -> bool:
    """
    Determine if the batch should be split into micro-batches.
    
    Args:
        batch_size: The size of the incoming batch.
        
    Returns:
        True if batch should be split, False otherwise.
    """
    if not is_microbatch_enabled():
        return False
    
    micro_batch_size = get_micro_batch_size()
    # If micro_batch_size <= 0, don't split (disabled)
    if micro_batch_size <= 0:
        return False
    # Only split if batch is larger than micro-batch size
    return batch_size > micro_batch_size



def compute_micro_batch_ranges(batch_size: int) -> List[Tuple[int, int]]:
    """
    Compute the (start, end) ranges for each micro-batch.
    
    Args:
        batch_size: The total batch size.
        
    Returns:
        A list of (start, end) tuples for each micro-batch.
    """
    micro_batch_size = get_micro_batch_size()

    if batch_size <= 0:
        return []
    if micro_batch_size <= 0:
        # Safety fallback: treat as no-split.
        return [(0, batch_size)]
    
    ranges = []
    start = 0
    while start < batch_size:
        end = min(start + micro_batch_size, batch_size)
        ranges.append((start, end))
        start = end
    
    return ranges


def split_tensor_to_microbatches(
    tensor: torch.Tensor,
    dim: int = 0
) -> List[torch.Tensor]:
    """
    Split a tensor along the batch dimension into micro-batches.
    
    Args:
        tensor: The input tensor to split. Expected shape: [batch_size, ...].
        dim: The dimension to split along (default: 0, batch dimension).
        
    Returns:
        A list of tensor chunks, one per micro-batch.
    """
    if tensor is None:
        return [None]
    
    batch_size = tensor.shape[dim]
    
    if not should_split_batch(batch_size):
        # No splitting needed, return as single-element list
        return [tensor]
    
    ranges = compute_micro_batch_ranges(batch_size)
    
    chunks = []
    for start, end in ranges:
        if dim == 0:
            chunk = tensor[start:end]
        elif dim == 1:
            chunk = tensor[:, start:end]
        else:
            # Generic case using narrow
            chunk = tensor.narrow(dim, start, end - start)
        chunks.append(chunk)
    
    return chunks


def merge_microbatch_outputs(
    outputs: List[torch.Tensor],
    dim: int = 0
) -> torch.Tensor:
    """
    Merge micro-batch outputs back into a single tensor.
    
    Args:
        outputs: A list of output tensors from each micro-batch.
        dim: The dimension to concatenate along (default: 0, batch dimension).
        
    Returns:
        A single merged tensor.
    """
    if len(outputs) == 1:
        return outputs[0]
    
    # Filter out None values
    valid_outputs = [o for o in outputs if o is not None]
    if not valid_outputs:
        return None
    
    first = valid_outputs[0]
    if not torch.is_tensor(first):
        return torch.cat(valid_outputs, dim=dim)

    same_shape = all(
        torch.is_tensor(out)
        and out.ndim == first.ndim
        and all(
            out.shape[axis] == first.shape[axis]
            for axis in range(first.ndim)
            if axis != dim
        )
        for out in valid_outputs[1:]
    )
    if same_shape:
        return torch.cat(valid_outputs, dim=dim)

    max_shape = list(first.shape)
    for out in valid_outputs[1:]:
        if not torch.is_tensor(out) or out.ndim != first.ndim:
            raise ValueError("Incompatible tensors for micro-batch merge")
        for axis, size in enumerate(out.shape):
            if axis == dim:
                continue
            max_shape[axis] = max(max_shape[axis], int(size))

    padded_outputs = []
    for out in valid_outputs:
        padded_shape = list(out.shape)
        for axis, size in enumerate(max_shape):
            if axis == dim:
                continue
            padded_shape[axis] = size
        if list(out.shape) == padded_shape:
            padded_outputs.append(out)
            continue
        padded = torch.zeros(
            padded_shape,
            dtype=out.dtype,
            device=out.device,
        )
        slices = [slice(None)] * out.ndim
        for axis, size in enumerate(out.shape):
            if axis == dim:
                continue
            slices[axis] = slice(0, int(size))
        padded[tuple(slices)] = out
        padded_outputs.append(padded)

    return torch.cat(padded_outputs, dim=dim)


def merge_microbatch_keep_indices(
    outputs: List[torch.Tensor],
    dim: int = 0,
    pad_value: int = -1,
) -> Optional[torch.Tensor]:
    """
    Merge per-micro-batch keep_indices tensors.

    Speculative pruning can produce a different number of kept positions per
    micro-batch. Pad non-batch dimensions with -1 so the merged tensor keeps a
    valid mask compatible with restore_hidden_states().
    """
    valid_outputs = [o for o in outputs if o is not None]
    if not valid_outputs:
        return None
    if len(valid_outputs) == 1:
        return valid_outputs[0]

    first = valid_outputs[0]
    if not torch.is_tensor(first):
        return torch.cat(valid_outputs, dim=dim)

    same_shape = all(
        torch.is_tensor(out)
        and out.ndim == first.ndim
        and all(
            out.shape[axis] == first.shape[axis]
            for axis in range(first.ndim)
            if axis != dim
        )
        for out in valid_outputs[1:]
    )
    if same_shape:
        return torch.cat(valid_outputs, dim=dim)

    max_shape = list(first.shape)
    for out in valid_outputs[1:]:
        if not torch.is_tensor(out) or out.ndim != first.ndim:
            raise ValueError("Incompatible keep_indices tensors for micro-batch merge")
        for axis, size in enumerate(out.shape):
            if axis == dim:
                continue
            max_shape[axis] = max(max_shape[axis], int(size))

    padded_outputs = []
    for out in valid_outputs:
        padded_shape = list(out.shape)
        for axis, size in enumerate(max_shape):
            if axis == dim:
                continue
            padded_shape[axis] = size
        if list(out.shape) == padded_shape:
            padded_outputs.append(out)
            continue
        padded = torch.full(
            padded_shape,
            pad_value,
            dtype=out.dtype,
            device=out.device,
        )
        slices = [slice(None)] * out.ndim
        for axis, size in enumerate(out.shape):
            if axis == dim:
                continue
            slices[axis] = slice(0, int(size))
        padded[tuple(slices)] = out
        padded_outputs.append(padded)

    return torch.cat(padded_outputs, dim=dim)


def log_microbatch_split(
    logger: logging.Logger,
    batch_size: int,
    num_microbatches: int,
    component: str = ""
) -> None:
    """
    Log micro-batch splitting information.
    
    Args:
        logger: The logger to use.
        batch_size: Original batch size.
        num_microbatches: Number of micro-batches created.
        component: Optional component name for context.
    """
    micro_batch_size = get_micro_batch_size()
    if not mbpipe_info_logs_enabled():
        return
    context = f" ({component})" if component else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} Split{context}: "
        f"batch_size={batch_size} -> {num_microbatches} micro-batches "
        f"(micro_batch_size={micro_batch_size})"
    )


def log_microbatch_merge(
    logger: logging.Logger,
    num_microbatches: int,
    merged_batch_size: int,
    component: str = ""
) -> None:
    """
    Log micro-batch merging information.
    
    Args:
        logger: The logger to use.
        num_microbatches: Number of micro-batches merged.
        merged_batch_size: Final merged batch size.
        component: Optional component name for context.
    """
    if not mbpipe_info_logs_enabled():
        return
    context = f" ({component})" if component else ""
    logger.info(
        f"{MBPIPE_LOG_PREFIX} Merge{context}: "
        f"{num_microbatches} micro-batches -> batch_size={merged_batch_size}"
    )


