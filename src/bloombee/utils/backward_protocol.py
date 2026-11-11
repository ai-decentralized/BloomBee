"""Cached, worker-to-worker training backward protocol."""

import re

from bloombee.data_structures import CHAIN_DELIMITER, UID_DELIMITER

BACKWARD_CHAIN = "backward_chain"
BACKWARD_CHAIN_VERSION = "backward_chain_version"
BACKWARD_PROTOCOL_VERSION = 2
BACKWARD_CACHE_ID = "backward_cache_id"
BACKWARD_NAMESPACE = "backward_namespace"
MAX_BACKWARD_STAGES = 32


class BackwardProtocolError(RuntimeError):
    """A worker cannot execute the mandatory cached backward protocol."""


def chain_response_metadata():
    return {BACKWARD_CHAIN_VERSION: BACKWARD_PROTOCOL_VERSION}


def validate_backward_chain(route, uid, position=0):
    """Validate a complete reverse route of forward cache handles."""
    if not isinstance(route, (list, tuple)) or not 1 <= len(route) <= MAX_BACKWARD_STAGES:
        raise ValueError("Invalid backward chain length")
    if type(position) is not int or not 0 <= position < len(route):
        raise ValueError("Invalid backward chain position")
    previous_start = None
    prefix = None
    for stage in route:
        if not isinstance(stage, dict) or not isinstance(stage.get("peer_id"), str):
            raise ValueError("Invalid backward chain peer")
        if not isinstance(stage.get("uid"), str):
            raise ValueError("Invalid backward chain stage")
        if not re.fullmatch(r"[0-9a-f]{32}", stage.get(BACKWARD_CACHE_ID, "")):
            raise ValueError("Invalid backward cache ID")
        if not re.fullmatch(r"backward-\d+", stage.get(BACKWARD_NAMESPACE, "")):
            raise ValueError("Invalid backward cache namespace")
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
    if route[position]["uid"] != uid:
        raise ValueError("Backward chain does not match requested blocks")
