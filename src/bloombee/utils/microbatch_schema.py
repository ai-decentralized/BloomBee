"""
Micro-batch Schema Definitions for Cross-Stage Pipeline

This module defines the schema for micro-batch payloads, including:
- Required vs optional tensor fields
- Request-level fields that can be cached and reused
- Default value factories for missing fields
- Utilities for filling defaults from request context

Used by both push side (handler.py) and consume side (block_functions.py)
to ensure consistent handling of micro-batch data.
"""

import os
from typing import Any, Dict, Optional, Set, Tuple
import torch
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

MBPIPE_SCHEMA_PREFIX = "[MBPIPE_SCHEMA]"
_NO_CONTEXT_WARNING_EMITTED = False
_STRICT_SCHEMA_WARN = os.environ.get("BLOOMBEE_MBPIPE_SCHEMA_WARN", "0") == "1"


# =============================================================================
# Field Definitions
# =============================================================================

# Fields that MUST be present in every micro-batch push
REQUIRED_FIELDS: Set[str] = {
    "hidden_states",
    "keep_indices",
}

# Fields that are request-level and only need to be sent with mb0
# These are cached and reused for subsequent micro-batches
REQUEST_LEVEL_FIELDS: Set[str] = {
    "prompts",
    "hypo_ids",
    "tree_attention_mask",
    "kv_cache_position_ids",
    "draft_tokens",
    "prefill_length",
    "is_spec_dec",
    "need_pruning",
}

# Metadata fields that should be carried with each micro-batch
MICROBATCH_METADATA_FIELDS: Set[str] = {
    "mb_idx",
    "expected_num_mb",
    "micro_batch_offset",
    "micro_batch_size",
    "full_batch_size",
    "request_id",
    "step_id",
    "session_id",
}


# =============================================================================
# Default Value Factories
# =============================================================================

def get_default_prompts(batch_size: int, num_backends: int) -> list:
    """Return default prompts (None for each backend)."""
    return [None] * num_backends


def get_default_hypo_ids(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return default hypo_ids (identity mapping)."""
    return torch.arange(batch_size, dtype=torch.int64, device=device)


def get_default_tree_attention_mask() -> None:
    """Return default tree_attention_mask (None - no speculative decoding)."""
    return None


def get_default_kv_cache_position_ids() -> None:
    """Return default kv_cache_position_ids (None)."""
    return None


def get_default_draft_tokens() -> None:
    """Return default draft_tokens (None - no speculative decoding)."""
    return None


def get_default_prefill_length() -> int:
    """Return default prefill_length (0)."""
    return 0


def get_default_is_spec_dec() -> bool:
    """Return default is_spec_dec (False)."""
    return False


def get_default_need_pruning() -> bool:
    """Return default need_pruning (False)."""
    return False


def get_default_keep_indices(seq_length: int, device: torch.device) -> torch.Tensor:
    """Return default keep_indices (full sequence)."""
    return torch.arange(seq_length, dtype=torch.int64, device=device)


# =============================================================================
# Request Context Management
# =============================================================================

class RequestContext:
    """
    Holds request-level data cached from the first micro-batch (mb0).
    
    This context is shared across all micro-batches of the same request,
    allowing mb1, mb2, ... to reuse fields sent only with mb0.
    """
    
    def __init__(self, request_id: str, step_id: Optional[str] = None):
        self.request_id = request_id
        self.step_id = step_id
        self.cached_fields: Dict[str, Any] = {}
        self.num_backends: int = 0
        self.is_initialized: bool = False
        
    def cache_from_mb0(
        self,
        prompts: Any = None,
        hypo_ids: Optional[torch.Tensor] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        prefill_length: int = 0,
        is_spec_dec: bool = False,
        need_pruning: bool = False,
        num_backends: int = 0,
    ) -> None:
        """Cache request-level fields from the first micro-batch."""
        self.cached_fields = {
            "prompts": prompts,
            "hypo_ids": hypo_ids,
            "tree_attention_mask": tree_attention_mask,
            "kv_cache_position_ids": kv_cache_position_ids,
            "draft_tokens": draft_tokens,
            "prefill_length": prefill_length,
            "is_spec_dec": is_spec_dec,
            "need_pruning": need_pruning,
        }
        self.num_backends = num_backends
        self.is_initialized = True
        logger.debug(
            f"{MBPIPE_SCHEMA_PREFIX} Cached request-level fields for "
            f"request={self.request_id}, step={self.step_id}"
        )
        
    def get_field(self, field_name: str, default: Any = None) -> Any:
        """Get a cached field value, or return default if not cached."""
        return self.cached_fields.get(field_name, default)
    
    def has_field(self, field_name: str) -> bool:
        """Check if a field has been cached."""
        return field_name in self.cached_fields


# =============================================================================
# Fill Defaults Utility
# =============================================================================

def fill_microbatch_defaults(
    mb_hidden_states: torch.Tensor,
    mb_keep_indices: Optional[torch.Tensor],
    request_context: Optional[RequestContext],
    num_backends: int,
) -> Tuple[
    torch.Tensor,  # hidden_states
    torch.Tensor,  # keep_indices
    Any,           # prompts
    torch.Tensor,  # hypo_ids
    Any,           # tree_attention_mask
    Any,           # kv_cache_position_ids
    Any,           # draft_tokens
    int,           # prefill_length
    bool,          # is_spec_dec
    bool,          # need_pruning
]:
    """
    Fill missing fields for a micro-batch using cached request context or defaults.
    
    Args:
        mb_hidden_states: The micro-batch hidden states tensor
        mb_keep_indices: The micro-batch keep indices (may be None)
        request_context: Cached request-level fields from mb0
        num_backends: Number of transformer backends
        
    Returns:
        Tuple of all required fields with defaults filled in
    """
    batch_size = mb_hidden_states.shape[0]
    seq_length = mb_hidden_states.shape[1]
    device = mb_hidden_states.device
    
    # Fill keep_indices
    if mb_keep_indices is None:
        keep_indices = get_default_keep_indices(seq_length, device)
        logger.debug(f"{MBPIPE_SCHEMA_PREFIX} Using default keep_indices (full seq)")
    else:
        keep_indices = mb_keep_indices
    
    # Fill from request context if available, otherwise use defaults
    if request_context is not None and request_context.is_initialized:
        prompts = request_context.get_field("prompts", get_default_prompts(batch_size, num_backends))
        hypo_ids = request_context.get_field("hypo_ids", get_default_hypo_ids(batch_size, device))
        tree_attention_mask = request_context.get_field("tree_attention_mask", get_default_tree_attention_mask())
        kv_cache_position_ids = request_context.get_field("kv_cache_position_ids", get_default_kv_cache_position_ids())
        draft_tokens = request_context.get_field("draft_tokens", get_default_draft_tokens())
        prefill_length = request_context.get_field("prefill_length", get_default_prefill_length())
        is_spec_dec = request_context.get_field("is_spec_dec", get_default_is_spec_dec())
        need_pruning = request_context.get_field("need_pruning", get_default_need_pruning())
    else:
        # No context available - use all defaults.
        # Default to debug-level to avoid noisy WARNs in decode loops;
        # enable strict warning with BLOOMBEE_MBPIPE_SCHEMA_WARN=1.
        global _NO_CONTEXT_WARNING_EMITTED
        if not _NO_CONTEXT_WARNING_EMITTED:
            msg = (
                f"{MBPIPE_SCHEMA_PREFIX} No request context available, using all defaults "
                f"(shown once; set BLOOMBEE_MBPIPE_SCHEMA_WARN=1 for WARN level)"
            )
            if _STRICT_SCHEMA_WARN:
                logger.warning(msg)
            else:
                logger.debug(msg)
            _NO_CONTEXT_WARNING_EMITTED = True
        else:
            logger.debug(f"{MBPIPE_SCHEMA_PREFIX} No request context available, using all defaults")
        prompts = get_default_prompts(batch_size, num_backends)
        hypo_ids = get_default_hypo_ids(batch_size, device)
        tree_attention_mask = get_default_tree_attention_mask()
        kv_cache_position_ids = get_default_kv_cache_position_ids()
        draft_tokens = get_default_draft_tokens()
        prefill_length = get_default_prefill_length()
        is_spec_dec = get_default_is_spec_dec()
        need_pruning = get_default_need_pruning()
    
    return (
        mb_hidden_states,
        keep_indices,
        prompts,
        hypo_ids,
        tree_attention_mask,
        kv_cache_position_ids,
        draft_tokens,
        prefill_length,
        is_spec_dec,
        need_pruning,
    )


# =============================================================================
# Micro-batch Queue Item Structure
# =============================================================================

def create_microbatch_queue_item(
    request_id: str,
    step_id: Optional[str],
    mb_idx: int,
    expected_num_mb: int,
    payload: Any,
    metadata: Dict[str, Any],
    offset: int,
    size: int,
    full_batch_size: int,
) -> Dict[str, Any]:
    """
    Create a standardized queue item for a micro-batch.
    
    This structure is used when putting micro-batches into the session queue
    for immediate consumption by the inference pipeline.
    """
    return {
        "type": "micro_batch",
        "request_id": request_id,
        "step_id": step_id,
        "mb_idx": mb_idx,
        "expected_num_mb": expected_num_mb,
        "payload": payload,
        "metadata": metadata,
        "offset": offset,
        "size": size,
        "full_batch_size": full_batch_size,
    }


def is_microbatch_queue_item(item: Any) -> bool:
    """Check if a queue item is a micro-batch (vs full batch request)."""
    if isinstance(item, dict):
        return item.get("type") == "micro_batch"
    return False


def create_microbatch_result_metadata(
    request_id: str,
    mb_idx: int,
    expected_num_mb: int,
    step_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create metadata for a micro-batch result."""
    return {
        "is_microbatch_result": True,
        "request_id": request_id,
        "mb_idx": mb_idx,
        "expected_num_mb": expected_num_mb,
        "step_id": step_id,
    }
