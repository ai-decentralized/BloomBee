"""Opt-in stateless LoRA RPCs, executed on the existing serial worker Runtime."""

import os
from contextlib import nullcontext
from typing import AsyncIterator

import torch
from async_timeout import timeout
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.streaming import split_for_streaming

from bloombee.utils.hivemind_compat import MSGPackSerializer, P2PContext, ServicerBase
from bloombee.utils.lora import LORA_NAMESPACE, LORA_PROTOCOL_VERSION, LoRAConfig, lora_context, projection_shapes
from bloombee.utils.lossless_transport import serialize_torch_tensor


def lora_enabled():
    return os.getenv("BLOOMBEE_ENABLE_LORA", "0") == "1"


def run_lora_block(backend, inputs, grad_outputs, metadata, *parameters):
    """Recompute a frozen block and return dX, dA, dB without accumulating worker .grad."""
    config = LoRAConfig(**metadata["config"])
    shapes = projection_shapes(backend.config)
    if len(parameters) != 2 * len(config.target_modules):
        raise ValueError("LoRA parameter count mismatch")
    if len(backend.module.module_shards) != 1:
        raise ValueError("LoRA does not yet support tensor-parallel blocks")
    for j, target in enumerate(config.target_modules):
        out_features, in_features = shapes[target]
        a, b = parameters[2 * j : 2 * j + 2]
        if a.shape != (config.r, in_features) or b.shape != (out_features, config.r):
            raise ValueError(f"Invalid LoRA A/B shapes for {target}")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("LoRA optimizer/master parameters must be float32")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("LoRA parameters must be finite")
    if inputs.ndim != 3 or inputs.shape[-1] != backend.config.hidden_size or not inputs.is_floating_point():
        raise ValueError("LoRA inputs must be [batch, sequence, hidden_size]")
    backward = grad_outputs.numel() != 0
    if backward and grad_outputs.shape != inputs.shape:
        raise ValueError("LoRA output gradient shape mismatch")
    backend._ensure_model_on_device()
    block = backend.module.module_shards[0]
    if backend.config.model_type == "llama":
        compute_dtype = block.weight_home[0].val[0].dtype
    else:
        compute_dtype = next(block.parameters()).dtype
    # Wire/client precision is independent of the actual frozen projection dtype.
    x = inputs.detach().to(dtype=compute_dtype).requires_grad_(backward)
    params = tuple(p.detach().requires_grad_(backward) for p in parameters)
    # HF offload restores Parameter.data to its CPU home before autograd runs.
    # Retain separate tensor handles to the staged storage for this one block's
    # backward; detach shares storage without following later .data reassignment.
    saved_weights = (
        torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t)
        if backward and getattr(backend.module, "_cpu_offload", False)
        else nullcontext()
    )
    # Override any outer runtime no_grad. Base parameters remain frozen.
    with saved_weights, torch.set_grad_enabled(backward), lora_context(
        config, params, training=metadata["training"], seed=metadata["seed"]
    ):
        output = backend.module(x)
        output = output[0] if isinstance(output, tuple) else output
        if not backward:
            return (output.detach().to(dtype=inputs.dtype),)
        gradients = torch.autograd.grad(output, (x, *params), grad_outputs.to(output), allow_unused=False)
    return (gradients[0].detach().to(dtype=inputs.dtype), *(g.detach() for g in gradients[1:]))


class LoRAService(ServicerBase):
    """No server-side adapter state: duplicate/retried calls cannot double-apply an update."""

    def __init__(self, handler):
        self.handler = handler

    async def rpc_lora_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        async with timeout(self.handler.request_timeout):
            uid, tensors, metadata = await self.handler._gather_inputs(requests, context)
            if metadata.get("version") != LORA_PROTOCOL_VERSION:
                raise ValueError("Unsupported LoRA protocol")
            config = LoRAConfig(**metadata["config"])
            names = uid.split(" ")
            if not names or len(set(names)) != len(names):
                raise ValueError("Invalid LoRA layer chain")
            indices = [int(name.rsplit(".", 1)[1]) for name in names]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError("LoRA layers must be a contiguous ordered span")
            backends = [self.handler.module_backends[name] for name in names]
            if any(not hasattr(b, "lora_pool") for b in backends):
                raise RuntimeError("Enable BLOOMBEE_ENABLE_LORA=1 on every worker")
            backward = metadata.get("backward")
            if not isinstance(backward, bool) or not isinstance(metadata.get("training"), bool):
                raise ValueError("LoRA phase and training must be booleans")
            seed = metadata.get("seed")
            if not isinstance(seed, int) or not 0 <= seed < 2**63:
                raise ValueError("Invalid LoRA recompute seed")
            width = 2 * len(config.target_modules)
            if len(tensors) != 1 + width * len(names) + int(backward):
                raise ValueError("LoRA request tensor count mismatch")
            x, parameters = tensors[0], tensors[1 : 1 + width * len(names)]
            if x.ndim != 3 or min(x.shape) < 1 or (backward and tensors[-1].shape != x.shape):
                raise ValueError("LoRA requires nonempty [batch, sequence, hidden] inputs and matching gradients")
            empty = torch.empty(0)
            inputs = [x]
            # During backward, recompute each layer input using the exact saved parameters and seeds.
            for j, backend in enumerate(backends[: len(backends) - int(backward)]):
                meta = {**metadata, "seed": seed + j}
                result = await backend.lora_pool.submit_task(
                    inputs[-1], empty, meta, *parameters[j * width : (j + 1) * width], priority=2.0
                )
                inputs.append(result[0])
            if backward:
                grad = tensors[-1]
                parameter_grads = [None] * len(backends)
                for j in reversed(range(len(backends))):
                    result = await backends[j].lora_pool.submit_task(
                        inputs[j],
                        grad,
                        {**metadata, "seed": seed + j},
                        *parameters[j * width : (j + 1) * width],
                        priority=2.0,
                    )
                    grad, parameter_grads[j] = result[0], result[1:]
                outputs = (grad, *(g for layer in parameter_grads for g in layer))
            else:
                outputs = (inputs[-1],)
            for tensor in outputs:
                wire = serialize_torch_tensor(tensor.detach().cpu().contiguous(), runtime_pb2.CompressionType.NONE)
                for part in split_for_streaming(wire, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(
                        tensors=[part], metadata=MSGPackSerializer.dumps({"version": LORA_PROTOCOL_VERSION})
                    )
