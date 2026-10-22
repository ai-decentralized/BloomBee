"""Worker-to-worker backward for frozen transformer blocks, without retained state."""

import torch

from bloombee.client.config import ClientConfig
from bloombee.client.remote_forward_backward import run_remote_backward_chain
from bloombee.server.block_functions import run_rpc_backward
from bloombee.utils.backward_protocol import BACKWARD_CHAIN, BACKWARD_CHAIN_VERSION, validate_backward_chain
from bloombee.utils.hivemind_compat import PeerID, get_logger

logger = get_logger(__name__)


async def run_backward_chain(handler, uid, tensors, backends, metadata):
    route = metadata[BACKWARD_CHAIN]
    if metadata.get(BACKWARD_CHAIN_VERSION) != 1:
        raise ValueError("Unsupported backward chain version")
    validate_backward_chain(route, uid, len(tensors))
    if route[0]["peer_id"] != str(handler._p2p.peer_id):
        raise ValueError("Backward chain reached an unexpected peer")
    if metadata.get("active_adapter") or any(getattr(backend, "optimizer", None) is not None for backend in backends):
        raise ValueError("Backward chaining requires frozen backbones without worker optimizers")
    inputs, grad_outputs, prompts, *remaining = tensors
    if inputs.ndim != 3 or inputs.shape != grad_outputs.shape:
        raise ValueError("Backward chain input and gradient shapes must match")
    grads = await run_rpc_backward(
        inputs,
        grad_outputs,
        prompts,
        requested_backends=backends,
        prioritizer=handler._prioritizer,
        active_adapter="",
        points=metadata.get("points", 0),
    )
    if len(route) == 1:
        logger.info("[S2S_BACKWARD] completed uid=%s", uid)
        return grads

    next_stage = route[1]
    next_peer = PeerID.from_base58(next_stage["peer_id"])
    stub = handler.get_stub(handler._p2p, next_peer)
    next_metadata = dict(next_stage.get("metadata", {}))
    next_metadata.update({BACKWARD_CHAIN: route[1:], BACKWARD_CHAIN_VERSION: 1})
    # Only the intermediate gradient goes directly to the preceding worker.
    # Inputs/prompts for the remaining stages were supplied by the client once.
    next_inputs, next_prompts, *rest = remaining
    logger.info(
        "[S2S_BACKWARD] uid=%s -> peer=%s uid=%s gradient_bytes=%d",
        uid,
        next_stage["peer_id"],
        next_stage["uid"],
        grads[0].numel() * grads[0].element_size(),
    )
    upstream_grads = await run_remote_backward_chain(
        next_stage["uid"],
        stub,
        (next_inputs, grads[0], next_prompts, *rest),
        config=ClientConfig(request_timeout=handler.request_timeout, connect_timeout=handler.request_timeout),
        metadata=next_metadata,
        compression=backends[0].args_schema[0].compression,
    )
    if len(grads) != len(upstream_grads) or len(grads) not in (1, 2):
        raise ValueError("Backward chain mixed prompt and non-prompt gradient schemas")
    if len(grads) == 2:
        # Return deep-prompt gradients in forward layer order, A before B.
        return [upstream_grads[0], torch.cat((upstream_grads[1], grads[1]), dim=0)]
    return upstream_grads
