"""Versioned, stateless routing for frozen-backbone backward RPCs."""

from bloombee.data_structures import CHAIN_DELIMITER, UID_DELIMITER

BACKWARD_CHAIN = "backward_chain"
BACKWARD_CHAIN_VERSION = "backward_chain_version"
MAX_BACKWARD_STAGES = 32


def validate_backward_chain(route, uid, tensor_count):
    """Require a finite, contiguous reverse route and one input/prompt per stage."""
    if not isinstance(route, (list, tuple)) or not 1 <= len(route) <= MAX_BACKWARD_STAGES:
        raise ValueError("Invalid backward chain length")
    if tensor_count != 2 * len(route) + 1:
        raise ValueError("Backward chain requires one gradient and an input/prompt pair per stage")
    previous_start = None
    prefix = None
    for stage in route:
        if not isinstance(stage, dict) or not isinstance(stage.get("peer_id"), str):
            raise ValueError("Invalid backward chain peer")
        if not isinstance(stage.get("uid"), str) or not isinstance(stage.get("metadata", {}), dict):
            raise ValueError("Invalid backward chain stage")
        metadata = stage.get("metadata", {})
        if metadata.get("active_adapter") or BACKWARD_CHAIN in metadata or BACKWARD_CHAIN_VERSION in metadata:
            raise ValueError("Backward chaining only supports frozen-backbone prompt training")
        indices = []
        for block_uid in stage["uid"].split(CHAIN_DELIMITER):
            block_prefix, index = block_uid.rsplit(UID_DELIMITER, 1)
            if prefix is None:
                prefix = block_prefix
            if block_prefix != prefix:
                raise ValueError("Backward chain mixes model prefixes")
            indices.append(int(index))
        if indices[0] < 0 or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Backward chain stage is not contiguous")
        if previous_start is not None and indices[-1] + 1 != previous_start:
            raise ValueError("Backward chain stages are not in reverse layer order")
        previous_start = indices[0]
    if route[0]["uid"] != uid:
        raise ValueError("Backward chain does not match requested blocks")


def chain_response_metadata():
    return {BACKWARD_CHAIN_VERSION: 1}
