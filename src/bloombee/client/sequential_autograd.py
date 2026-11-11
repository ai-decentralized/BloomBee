"""
A PyTorch autograd function that runs forward/backward on a sequence of remote servers in a fault-tolerant manner
"""
import asyncio
import itertools
import uuid
from collections import deque
from typing import List, Optional, Sequence, Tuple

import torch
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

from bloombee.client.remote_forward_backward import run_remote_backward_chain, run_remote_forward
from bloombee.client.routing import RemoteSequenceManager, maybe_log_traceback
from bloombee.data_structures import CHAIN_DELIMITER, RemoteSpanInfo
from bloombee.server.backward_chain import BackwardService
from bloombee.server.handler import TransformerConnectionHandler
from bloombee.utils.backward_protocol import (
    BACKWARD_CACHE_ID,
    BACKWARD_CHAIN,
    BACKWARD_CHAIN_VERSION,
    BACKWARD_NAMESPACE,
    BACKWARD_PROTOCOL_VERSION,
    BackwardProtocolError,
    validate_backward_chain,
)
from bloombee.utils.hivemind_compat import MSGPackSerializer
from bloombee.utils.misc import DUMMY, is_dummy

logger = get_logger(__name__)

MAX_TOKENS_IN_BATCH = 1024


async def sequential_forward(
    inputs: torch.Tensor,
    prompts: torch.Tensor,
    sequence_manager: RemoteSequenceManager,
    start_index: int = 0,
    end_index: Optional[int] = None,
    cache_for_backward: bool = True,
) -> Tuple[torch.Tensor, Sequence[dict], Sequence[RemoteSpanInfo]]:
    """
    Constructs a routing path from <start_index> to <end_index>.
    Performs chained forward for each subsequence of blocks on the path.
    If some subsequence fails, reconstructs the remaining path and tries to finish the forward.
    """

    assert isinstance(inputs, torch.Tensor) and inputs.ndim == 3, f"{type(inputs)}: {inputs.ndim}"

    inputs_device = inputs.device
    inputs_dtype = inputs.dtype
    inputs = inputs.cpu()
    prompts = prompts.cpu()

    end_index = end_index if end_index is not None else len(sequence_manager.block_uids)
    assert start_index >= 0 and end_index <= len(sequence_manager.block_uids)
    assert is_dummy(prompts) or len(prompts) == len(
        sequence_manager.block_uids
    )  # should be n_layers - 1 but add extra prompts for convenience

    sequences = deque()
    backward_records = []
    done_sequences = []

    block_idx = start_index
    while block_idx < end_index:
        for attempt_no in itertools.count():
            logger.debug(f"Forward: block {block_idx}, attempt {attempt_no}")
            span = None
            try:
                if not sequences or attempt_no >= 1:
                    sequences = deque(sequence_manager.make_sequence(block_idx, end_index, mode="max_throughput"))
                    # make_sequence() could return a longer sequence
                    sequences[-1].end = min(sequences[-1].end, end_index)
                    logger.debug(f"Found path from block {block_idx} to {end_index} via {len(sequences)} servers")

                span = sequences.popleft()

                stub = TransformerConnectionHandler.get_stub(sequence_manager.state.p2p, span.peer_id)
                flat_tensors = (inputs, prompts[span.start : span.end])

                span_uids = CHAIN_DELIMITER.join(sequence_manager.block_uids[span.start : span.end])
                metadata = dict(sequence_manager.get_request_metadata("rpc_forward", span_uids, *flat_tensors) or {})
                response_metadata = {}
                if cache_for_backward:
                    metadata.update(
                        {BACKWARD_CACHE_ID: uuid.uuid4().hex, BACKWARD_CHAIN_VERSION: BACKWARD_PROTOCOL_VERSION}
                    )
                (outputs,) = await run_remote_forward(
                    span_uids,
                    stub,
                    sequence_manager.rpc_info,
                    *flat_tensors,
                    config=sequence_manager.config,
                    metadata=MSGPackSerializer.dumps(metadata),
                    response_metadata=response_metadata,
                )

                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape == inputs.shape, f"Expected output {inputs.shape}, got {outputs.shape}"

                if cache_for_backward:
                    if response_metadata.get(BACKWARD_CHAIN_VERSION) != BACKWARD_PROTOCOL_VERSION:
                        raise BackwardProtocolError(
                            "Worker does not support cached backward; upgrade all model workers"
                        )
                    if response_metadata.get(BACKWARD_CACHE_ID) != metadata[BACKWARD_CACHE_ID]:
                        raise BackwardProtocolError("Worker acknowledged a different forward cache")
                    backward_records.append(
                        {
                            "uid": span_uids,
                            "peer_id": str(span.peer_id),
                            BACKWARD_CACHE_ID: metadata[BACKWARD_CACHE_ID],
                            BACKWARD_NAMESPACE: response_metadata[BACKWARD_NAMESPACE],
                        }
                    )
                done_sequences.append(span)

                inputs = outputs
                block_idx = span.end
                sequence_manager.on_request_success(span.peer_id)
                break
            except Exception as e:
                if isinstance(e, BackwardProtocolError):
                    raise
                sequence_manager.on_request_failure(span.peer_id if span is not None else None)
                if attempt_no + 1 == sequence_manager.config.max_retries:
                    raise
                delay = sequence_manager.get_retry_delay(attempt_no)
                logger.warning(
                    f"Caught exception when running forward via {span} (retry in {delay:.0f} sec): {repr(e)}"
                )
                maybe_log_traceback(e)
                await asyncio.sleep(delay)

    outputs = inputs.to(device=inputs_device, dtype=inputs_dtype)
    return outputs, backward_records, done_sequences


async def sequential_backward(grad_outputs, backward_records, prompts, forward_sequences, sequence_manager):
    """Start the reverse worker chain and receive its result directly from stage A."""
    route = list(reversed(backward_records))
    if not route or len(route) != len(forward_sequences):
        raise RuntimeError("Backward requires the forward cache handles for every stage")
    validate_backward_chain(route, route[0]["uid"])
    config, p2p = sequence_manager.config, sequence_manager.state.p2p
    metadata = {
        BACKWARD_CHAIN: route,
        BACKWARD_CHAIN_VERSION: BACKWARD_PROTOCOL_VERSION,
        "client_peer": str(p2p.peer_id),
        "position": 0,
    }

    def stub(stage):
        from bloombee.utils.hivemind_compat import PeerID

        return BackwardService.get_stub(p2p, PeerID.from_base58(stage["peer_id"]), namespace=stage[BACKWARD_NAMESPACE])

    compression = sequence_manager.rpc_info["forward_schema"][0][0].compression
    tasks = [
        asyncio.create_task(
            run_remote_backward_chain(
                route[-1]["uid"],
                stub(route[-1]),
                (),
                config=config,
                metadata={**metadata, "position": len(route) - 1},
                compression=compression,
                collect=True,
            )
        ),
        asyncio.create_task(
            run_remote_backward_chain(
                route[0]["uid"],
                stub(route[0]),
                grad_outputs,
                config=config,
                metadata=metadata,
                compression=compression,
            )
        ),
    ]
    completed = False
    try:
        grads, ack = await asyncio.gather(*tasks)
        expected = 1 if is_dummy(prompts) else 2
        if ack or len(grads) != expected or grads[0].shape != grad_outputs[0].shape:
            raise RuntimeError("Worker chain returned an invalid gradient schema")
        if expected == 2 and grads[1].shape != prompts.shape:
            raise RuntimeError("Worker chain returned an invalid deep-prompt gradient shape")
        for span in forward_sequences:
            sequence_manager.on_request_success(span.peer_id)
        logger.info(
            "[S2S_BACKWARD_CLIENT] stages=%d activation_upload_bytes=0 result_direct_from_first_worker=1", len(route)
        )
        completed = True
        return [grads[0].to(grad_outputs[0])], grads[1].to(prompts) if expected == 2 else None
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not completed:
            # Release unconsumed caches on errors as well; never reroute via Client.
            await asyncio.gather(
                *[
                    stub(stage).rpc_release(
                        runtime_pb2.ExpertRequest(
                            uid=stage["uid"],
                            metadata=MSGPackSerializer.dumps({BACKWARD_CACHE_ID: stage[BACKWARD_CACHE_ID]}),
                        ),
                        timeout=config.connect_timeout,
                    )
                    for stage in route
                ],
                return_exceptions=True,
            )
        backward_records.clear()
        forward_sequences.clear()


async def _gather_forward(input_batches, prompt_batches, sequence_manager, cache_for_backward):
    """Wrapper for asyncio.gather to perform parallel sequential forwards"""
    return await asyncio.gather(
        *[
            sequential_forward(input_batch, prompt_batch, sequence_manager, cache_for_backward=cache_for_backward)
            for input_batch, prompt_batch in zip(input_batches, prompt_batches)
        ]
    )


async def _gather_backward(
    grad_output_batches, intermediate_input_batches, prompt_batches, forward_sequences, sequence_manager
):
    """Wrapper for asyncio.gather to perform parallel sequential backwards"""
    return await asyncio.gather(
        *[
            sequential_backward((grad_output,), input_batch, prompt_batch, spans, sequence_manager)
            for grad_output, input_batch, prompt_batch, spans in zip(
                grad_output_batches, intermediate_input_batches, prompt_batches, forward_sequences
            )
        ]
    )


class _RemoteSequentialAutogradFunction(torch.autograd.Function):
    """
    PyTorch autograd function that provides forward and backward calls for the entire sequence of remote transformer blocks.
    This function splits input data into batches with <MAX_TOKENS_IN_BATCH> and performs efficient parallel processing.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, prompts: torch.Tensor, sequence_manager: RemoteSequenceManager):
        batch_size = max(MAX_TOKENS_IN_BATCH // inputs.shape[1], 1)
        input_batches: Sequence[torch.Tensor] = inputs.detach().split(batch_size)
        if prompts is None or is_dummy(prompts):
            prompt_batches = [DUMMY] * len(input_batches)
        else:
            prompt_batches: Sequence[torch.Tensor] = prompts.detach().split(batch_size, dim=1)

        sequence_manager.rpc_info  # lazy init
        outputs = RemoteExpertWorker.run_coroutine(
            _gather_forward(input_batches, prompt_batches, sequence_manager, any(ctx.needs_input_grad[:2]))
        )
        assert len(outputs) == len(input_batches)

        output_batches = [output[0] for output in outputs]
        intemediate_input_batches = [output[1] for output in outputs]
        sequences_for_batches = [output[2] for output in outputs]

        ctx.prompt_batches = prompt_batches
        ctx.sequence_manager = sequence_manager
        ctx.backward_records_for_batches = intemediate_input_batches
        ctx.sequences_for_batches = sequences_for_batches
        return torch.cat(output_batches, dim=0)

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        intermediate_input_batches: List[Sequence[torch.Tensor]] = ctx.backward_records_for_batches
        forward_sequences: List[Sequence[RemoteSpanInfo]] = ctx.sequences_for_batches
        ctx.sequence_manager.rpc_info  # lazy init

        batch_size = max(MAX_TOKENS_IN_BATCH // grad_outputs.shape[1], 1)
        grad_output_batches: Sequence[torch.Tensor] = grad_outputs.split(batch_size)
        assert len(intermediate_input_batches) == len(grad_output_batches) == len(forward_sequences)

        outputs = RemoteExpertWorker.run_coroutine(
            _gather_backward(
                grad_output_batches,
                intermediate_input_batches,
                ctx.prompt_batches,
                forward_sequences,
                ctx.sequence_manager,
            )
        )
        grad_input_batches = [output[0][0] for output in outputs]
        grad_prompt_batches = [output[1] for output in outputs]

        grad_inputs = torch.cat(grad_input_batches, dim=0)
        dummy_grad_prompts = [grad_prompt is None for grad_prompt in grad_prompt_batches]
        grad_prompts = torch.cat(grad_prompt_batches, dim=1) if not any(dummy_grad_prompts) else None
        return (grad_inputs, grad_prompts, None)
