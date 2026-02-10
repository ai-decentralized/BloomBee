from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import torch
from hivemind import MSGPackSerializer, anext, get_logger
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.client.config import ClientConfig
from bloombee.client.routing import RemoteSequenceManager, maybe_log_traceback
from bloombee.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from bloombee.server.handler import TransformerConnectionHandler
from bloombee.utils.lossless_transport import deserialize_torch_tensor, serialize_torch_tensor
from bloombee.utils.misc import DUMMY, DUMMY_INT64, is_dummy
from bloombee.utils.packaging import pack_args_kwargs, normalize_arg
from bloombee.utils.microbatch_config import (
    is_microbatch_enabled,
    get_micro_batch_size,
    get_current_path,
    log_config as mbpipe_log_config,
    log_path_entry as mbpipe_log_path_entry,
    MBPIPE_LOG_PREFIX,
)

logger = get_logger(__name__)


class _ServerInferenceSession:
    """
    An interface to a single multi-step *inference* session for a a set of blocks on a specific server.

    :note: This class is *not* fault-tolerant out of the box.
    """

    def __init__(
        self,
        config: ClientConfig,
        span: RemoteSpanInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        inputs_queue: asyncio.Queue,
        outputs_aiter: AsyncIterator,
        *,
        max_length: int,
        **metadata,
    ):
        self.config = config
        self.span, self.uid, self.rpc_info = span, uid, rpc_info
        self.num_blocks = uid.count(CHAIN_DELIMITER) + 1
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.session_id = str(uuid.uuid4())
        self.session_metadata = dict(max_length=max_length, **metadata)
        self.stepped = False
        self.closed = False

        self._position = 0
        self.history = None  # Used in case of server failures to regenerate attention caches on new servers
        self.next_session = None

    @classmethod
    async def create(
        cls,
        config: ClientConfig,
        p2p: P2P,
        span: RemoteSpanInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        **metadata,
    ) -> _ServerInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        stub = TransformerConnectionHandler.get_stub(p2p, span.peer_id)
        inputs_queue = asyncio.Queue()
        outputs_stream = await asyncio.wait_for(
            stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
            config.connect_timeout,
        )
        return cls(config, span, uid, rpc_info, inputs_queue, outputs_stream, **metadata)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, input_timeout: Optional[float] = None) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), input_timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, start_from_position: int):
        # assert start_from_position <= self._position
        self._position = start_from_position
        if self.history is not None and self.history.shape[1] >= start_from_position:
            self.history = self.history[:, :start_from_position, :] if start_from_position > 0 else None

    def step(
        self,
        inputs: torch.Tensor,
        prompts: torch.Tensor,
        hypo_ids: torch.LongTensor,
        tree_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        prefill_length: int = 0,
        keep_indices: Optional[torch.Tensor] = None,
        need_pruning: bool = False,
        is_spec_dec: bool = False,
        *,
        step_id: str,
    ) -> torch.Tensor:
        """
        Inference step: send a chunk of input tensors and receive a chunk of outputs
        :prompts: optional DEEP prompts, added to a prefix of each layer's outputs,
          if specified, deep prompts should have shape [num_layers, batch_size, prefix_len, hid_size]
        """
        if self.closed:
            raise Exception("Session is closed, cannot perform step")
        if is_spec_dec:
            n_input_tokens = 0 if kv_cache_position_ids is None else kv_cache_position_ids.numel()
        else:
            n_input_tokens = inputs.shape[1]
        # print('client step() n_input_tokens', n_input_tokens)
        if self.history is None: # if the history log is empty
            self.history = inputs # assign the current inputs to the history log
        elif self.history.shape[1] == self._position: # if the length of the history equals the current position
            self.history = torch.cat([self.history, inputs[:, -n_input_tokens:]], dim=1) # 将当前输入的最后n_input_tokens个token拼接到历史记录中
        # history can cat input if it's spec decoding and pruning happened, need fall  back
        # assert self.history.shape[1] == self._position + n_input_tokens,
        #     f"Broken input cache: span={self.span} shape={self.history.shape} "
        #     f"position={self._position} n_input_tokens={n_input_tokens}"
        # )

        if not self.stepped: # if not exe step yet
            inputs = self.history  # Pass full inputs including prefix
        else:
            inputs = inputs  # No need to pass prefix further

        def _infer_batch_dim(value) -> int:
            if value is None or is_dummy(value):
                return 0
            if torch.is_tensor(value):
                if value.ndim == 0:
                    return 1
                return int(value.shape[0]) if value.shape else 1
            try:
                return int(len(value))
            except Exception:
                return 0

        # For speculative decoding, hidden states may be pruned/compressed on some steps.
        # Derive a stable logical full-batch size from all request tensors and pass it
        # explicitly so server-side KV allocation stays consistent across the session.
        logical_full_batch_size = max(
            _infer_batch_dim(inputs),
            _infer_batch_dim(hypo_ids),
            _infer_batch_dim(keep_indices),
            _infer_batch_dim(prefill_length),
            _infer_batch_dim(draft_tokens),
            _infer_batch_dim(tree_attention_mask),
            1,
        )

        # serialize inputs and put them into the queue
        
        input_tensors, args_structure = pack_args_kwargs(
            inputs, 
            normalize_arg(keep_indices),
            normalize_arg(torch.tensor(1 if need_pruning else 0)),
            prompts, hypo_ids, 
            normalize_arg(tree_attention_mask),
            normalize_arg(kv_cache_position_ids),
            normalize_arg(draft_tokens),
            normalize_arg(prefill_length),
            normalize_arg(torch.tensor(1 if is_spec_dec else 0)),
        )
        logger.info(f"_ServerInferenceSession  step id {step_id}")
        request_metadata = dict(session_id=self.session_id, step_id=step_id)
        if not self.stepped:
            request_metadata.update(self.session_metadata)
        # Explicitly expose speculative-decoding mode in metadata so the server
        # can make safe KV allocation decisions before first step processing.
        request_metadata["is_spec_dec"] = 1 if is_spec_dec else 0
        request_metadata["full_batch_size"] = int(logical_full_batch_size)
        request_metadata["micro_batch_size"] = int(inputs.shape[0]) if inputs.ndim >= 1 else 1
        if is_spec_dec:
            request_metadata["start_from_position"] = self._position + n_input_tokens
        else:
            if self._position is not None:
                request_metadata["start_from_position"] = self._position
        # Enable server-to-server communication to trigger CROSS_GPU_TRANSFER
        # Speculative decoding keeps strict full-batch semantics; avoid cross-stage push.
        if self.config.use_server_to_server and not is_spec_dec:
            next_servers = self._collect_next_servers()
            if next_servers:
                request_metadata["next_servers"] = next_servers
        elif is_spec_dec:
            request_metadata["disable_cross_stage_push"] = 1

        request_metadata["args_structure"] = args_structure

        # TODO: make possible to use different compression method for different tensors
        server_side_inference_schema, kwargs_schema = self.rpc_info["inference_schema"]
        compression = server_side_inference_schema[0].compression
        inference_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in input_tensors)

        # TODO: create more explicit way to check servers schema and client's structure
        # assert len(input_tensors) >= len(
        #     server_side_inference_schema
        # ), "Hidden_state, prompts and hypo_ids tensors are necessary for an inference step"

        # [NETWORK_TIMING] Measure serialization time
        serialize_start = time.perf_counter()
        
        # Serialize and send data (debug output removed for performance)
        # Fix for bus error in cross-machine setups: ensure tensors are contiguous before serialization
        serialized_tensors = [
            serialize_torch_tensor(
                tensor.contiguous().to(proto.dtype) if not tensor.is_contiguous() else tensor.to(proto.dtype),
                proto.compression
            )
            for tensor, proto in zip(input_tensors, inference_schema)
        ]
        serialized_metadata = MSGPackSerializer.dumps(request_metadata)
        
        serialize_end = time.perf_counter()
        serialize_time_ms = (serialize_end - serialize_start) * 1000
        
        # [NETWORK_TIMING] Measure serialized data size
        total_tensor_bytes = sum(len(t.buffer) for t in serialized_tensors)
        metadata_bytes = len(serialized_metadata)
        total_send_bytes = total_tensor_bytes + metadata_bytes
        
        logger.info(f"[NETWORK_TX] SEND_START | step_id={step_id} | "
                   f"tensor_size={total_tensor_bytes/1024:.2f}KB | "
                   f"metadata_size={metadata_bytes}B | "
                   f"total={total_send_bytes/1024:.2f}KB | "
                   f"serialize_time={serialize_time_ms:.2f}ms")
        
        # [NETWORK_TIMING] Measure network round-trip time
        network_start = time.perf_counter()
        
        outputs_serialized = RemoteExpertWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=serialized_tensors,
                    metadata=serialized_metadata,
                )
            )
        )
        
        network_end = time.perf_counter()
        network_rtt_ms = (network_end - network_start) * 1000
        
        # [NETWORK_TIMING] Measure deserialization time
        deserialize_start = time.perf_counter()
        outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
        deserialize_end = time.perf_counter()
        deserialize_time_ms = (deserialize_end - deserialize_start) * 1000
        
        # [NETWORK_TIMING] Measure received data size
        total_recv_bytes = sum(len(t.buffer) for t in outputs_serialized.tensors)
        
        logger.info(f"[NETWORK_TX] RECV_END | step_id={step_id} | "
                   f"recv_size={total_recv_bytes/1024:.2f}KB | "
                   f"network_rtt={network_rtt_ms:.2f}ms | "
                   f"deserialize_time={deserialize_time_ms:.2f}ms")
        
        # [NETWORK_TIMING] Summary log
        total_time_ms = serialize_time_ms + network_rtt_ms + deserialize_time_ms
        logger.info(f"[NETWORK_TX] SUMMARY | step_id={step_id} | "
                   f"send={total_send_bytes/1024:.2f}KB | recv={total_recv_bytes/1024:.2f}KB | "
                   f"serialize={serialize_time_ms:.2f}ms | network={network_rtt_ms:.2f}ms | "
                   f"deserialize={deserialize_time_ms:.2f}ms | total={total_time_ms:.2f}ms")
        # assert (
        #     outputs[0].shape == inputs.shape
        # ), f"output activation shape is different from input shape: {outputs[0].shape} != {inputs.shape}"

        self._position += n_input_tokens
        logger.info(f"server inference session self._position: {self._position}")
        return outputs

    def _collect_next_servers(self) -> List[Tuple[str, str, int, int]]:
        next_servers = []
        session = self.next_session
        while session is not None and session.stepped: 
            next_servers.append(
                (session.span.peer_id.to_base58(), session.session_id, session.span.start, session.span.end)
            )
            session = session.next_session
        return next_servers

    async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
        """Inference step on serialized data. This code is meant to be run inside RemoteExpertWorker"""
        await self._inputs_queue.put(inputs_serialized)
        self.stepped = True
        return await asyncio.wait_for(anext(self._outputs_stream), self.config.request_timeout)

    def close(self):
        """Finish a given inference session, close the underlying connection"""
        if self._outputs_stream is None:
            return  # already closed
        RemoteExpertWorker.run_coroutine(self._aclose_stream())
        self._outputs_stream = self._inputs_queue = None
        self.closed = True

    async def _aclose_stream(self):
        """Close the inference session. This code is meant to be run inside RemoteExpertWorker"""
        if self._outputs_stream is None:
            return  # already closed
        if self.stepped:
            await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
            try:
                await anext(self._outputs_stream)
            except StopAsyncIteration:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, *exc_details):
        self.close()


class InferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, max_length: int):
        self._sequence_manager = sequence_manager
        self._closed = False
        self._server_sessions = []
        self._position = 0
        self._max_length = max_length
        self.output_ids = None
        self.past_key_values = None
        self.keep_indices = None
        self.prefill_length = 0
        self._step_count = 0  # Track step count for logging
        
        # [MBPIPE] Log micro-batch pipeline configuration at client session creation
        mbpipe_log_config(logger, context="InferenceSession.__init__")
        self.first_inference = True

    @property
    def num_blocks(self) -> int:
        return len(self._sequence_manager)

    @property
    def position(self) -> int:
        return self._position

    @position.setter 
    def position(self, start_from_position: int) -> None: # 设置一个位置属性，并确保所有相关的会话对象都同步更新这个位置。
        self._position = start_from_position # set a position attribute and ensure that all related session objects are updated to reflect this position synchronously.
        for session in self._server_sessions:
            assert isinstance(session, _ServerInferenceSession)
            session.position = start_from_position

    def _enter_server_sessions(self, chosen_spans: List[RemoteSpanInfo]) -> List[_ServerInferenceSession]:
        server_sessions = [] # 创建一组服务器会话，并在发生错误时确保已创建的会话能够正确退出。
        try:
            for span in chosen_spans:
                span_uids = CHAIN_DELIMITER.join(self._sequence_manager.block_uids[span.start : span.end])
                metadata = self._sequence_manager.get_request_metadata("rpc_inference", span_uids, peer_id=span.peer_id)
                session = RemoteExpertWorker.run_coroutine(
                    _ServerInferenceSession.create(
                        self._sequence_manager.config,
                        self._sequence_manager.state.p2p,
                        span,
                        span_uids,
                        rpc_info=self._sequence_manager.rpc_info,
                        max_length=self._max_length,
                        **metadata,
                    )
                )
                server_sessions.append(session)
                session.__enter__()
            return server_sessions
        except Exception:
            self._exit_server_sessions(server_sessions)
            raise

    def _exit_server_sessions(self, server_sessions: List[_ServerInferenceSession]) -> None:
        for session in reversed(server_sessions):
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug("Caught exception while closing connection to server:", exc_info=True)

    def __enter__(self) -> "InferenceSession":
        assert not self._closed and not self._server_sessions
        return self

    def step(   # 执行一次推理步骤，处理输入数据和相应的提示与假设 ID，同时在可能出现错误的情况下进行重试。
        self,
        inputs: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        hypo_ids: Optional[torch.Tensor] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        kv_cache_position_ids: Optional[torch.Tensor] = None,
        draft_tokens: Optional[torch.Tensor] = None,
        is_spec_decoding: Optional[torch.Tensor] = None,
        prefill_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not self._closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")

        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4, "deep prompts should have shape [num_blocks, batch_size, prefix_len, hid_size]"
            assert prompts.shape[0] == self.num_blocks
            assert prompts.shape[1] in (inputs.shape[0], 1)
            assert prompts.shape[2] <= inputs.shape[1]
            assert prompts.shape[3] == inputs.shape[2]

        if hypo_ids is None or is_dummy(hypo_ids):
            hypo_ids = DUMMY_INT64
        else:
            assert len(hypo_ids) == len(inputs)
            assert hypo_ids.dtype == torch.int64

        inputs_device = inputs.device
        inputs_dtype = inputs.dtype
        
        inputs = inputs.cpu()
        prompts = prompts.cpu()
        hypo_ids = hypo_ids.cpu()
        tree_attention_mask = tree_attention_mask.cpu() if tree_attention_mask is not None else None
        kv_cache_position_ids = kv_cache_position_ids.cpu() if kv_cache_position_ids is not None else None
        draft_tokens = draft_tokens.cpu() if draft_tokens is not None else None
        is_spec_decoding = is_spec_decoding.cpu() if is_spec_decoding is not None else None
        
        step_id = str(uuid.uuid4())  # Generate a unique step ID.
        
        # [MBPIPE] Log current path at client step entry (first step only to reduce noise)
        self._step_count += 1
        if self._step_count == 1:
            batch_size = inputs.shape[0] if inputs.ndim >= 1 else 1
            mbpipe_log_path_entry(logger, "client.InferenceSession.step", batch_size=batch_size)

        n_input_tokens = inputs.shape[1] if kv_cache_position_ids is None else kv_cache_position_ids.numel()
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {self._position} + current {n_input_tokens} exceeds pre-allocated maximum {self._max_length}"
            )

        server_idx = 0
        block_idx = 0
        inference_step_start = time.perf_counter()
        batch_size = inputs.shape[0] if inputs.ndim >= 1 else 1
        if tree_attention_mask is not None and prefill_length is not None:
            self.prefill_length = prefill_length.to(inputs.device)
        else:
            self.prefill_length = torch.zeros(batch_size, device=inputs.device)
        keep_indices = torch.arange(
            inputs.shape[1],
            dtype=torch.int64,
            device=inputs.device
        ).unsqueeze(0).expand(inputs.shape[0], -1)
        self.keep_indices = keep_indices
        if is_spec_decoding is not None and is_spec_decoding.item() == 1:
            is_spec_dec = True
        else:
            is_spec_dec = False
        need_pruning = is_spec_dec
        while block_idx < self.num_blocks:
            for attempt_no in itertools.count():
                logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
                server_session = None
                try:
                    if not self._server_sessions or attempt_no >= 1:
                        self._update_sequence(server_idx, block_idx, attempt_no)

                    server_session = self._server_sessions[server_idx]
                    # assert server_session.position == self.position, f"{server_session.position} and {self.position}"
                    
                    # 🔍 CLIENT DEBUG: Log server span processing start
                    span_start_time = time.perf_counter()
                    
                    inputs, keep_indices, need_pruning_next = server_session.step( 
                        inputs,
                        prompts[server_session.span.start : server_session.span.end],
                        hypo_ids,
                        tree_attention_mask,
                        kv_cache_position_ids,
                        draft_tokens,
                        self.prefill_length,
                        self.keep_indices,
                        need_pruning,
                        is_spec_dec,
                        step_id=step_id,
                    )
                    if is_spec_dec and need_pruning:
                        self.keep_indices = keep_indices
                    
                    need_pruning = False  # only need to prune on the first server
                    
                    # 🔍 CLIENT DEBUG: Log server span processing end
                    span_end_time = time.perf_counter()
                    span_duration = (span_end_time - span_start_time) * 1000  # ms
                    logger.info(f"[CLIENT_SERVER_END] ServerIdx={server_idx} | Blocks={server_session.span.start}:{server_session.span.end} | Duration={span_duration:.2f}ms")
                    # print('inputs ', inputs)
                    # print('inputs.shape ', inputs.shape)
                    server_idx += 1
                    block_idx = server_session.span.end
                    self._sequence_manager.on_request_success(server_session.span.peer_id)
                    break
                except Exception as e:
                    self._sequence_manager.on_request_failure(
                        server_session.span.peer_id if server_session is not None else None
                    )
                    if attempt_no + 1 == self._sequence_manager.config.max_retries:
                        raise
                    delay = self._sequence_manager.get_retry_delay(attempt_no)
                    logger.warning(
                        f"Caught exception when running inference via {server_session.span if server_session is not None else None} "
                        f"(retry in {delay:.0f} sec): {repr(e)}"
                    )
                    maybe_log_traceback(e)
                    time.sleep(delay) 

        self._position += n_input_tokens
        # logger.info(f"keep_indices: {keep_indices}")
        # logger.info(f"before _recover_hidden_states: {inputs}")
        if draft_tokens is not None and is_spec_dec:
            inputs = self._restore_hidden_states(inputs, self.keep_indices, draft_tokens.shape[1])
        # logger.info(f"after _recover_hidden_states: {inputs}")
        outputs = inputs 
        
        # 🔍 CLIENT DEBUG: Log inference step end
        inference_step_end = time.perf_counter()
        inference_step_duration = (inference_step_end - inference_step_start) * 1000  # ms
        logger.info(f"[CLIENT_INFERENCE_END] Position={self._position} | Duration={inference_step_duration:.2f}ms | Servers={server_idx}")
        logger.info("="*80)
        
        outputs = outputs.to(device=inputs_device, dtype=inputs_dtype) 
        # print('client inference session outputs ', outputs.shape)
        return outputs
    
    def _restore_hidden_states(
        self,
        flattened_hidden_states: torch.Tensor,  # [N_total_valid, H] or [B, L_keep/full, H]
        keep_indices: torch.Tensor,  # [B, max_keep_len]，padding 为 -1
        original_seq_len: int,  # 原始序列长度
    ) -> torch.Tensor:
        """
        将铺平的 hidden states 还原为 [B, original_seq_len, hidden_size]
        
        Args:
            flattened_hidden_states:
                - [N_total_valid, hidden_size] 铺平后的有效 hidden states
                - [B, max_keep_len, hidden_size] 每 batch 的压缩 hidden states
                - [B, original_seq_len, hidden_size] 已经是完整长度（将原样返回）
            keep_indices: [B, max_keep_len] 每个 batch 的 keep indices，padding 为 -1
            original_seq_len: 原始序列长度
        
        Returns:
            restored_hidden_states: [B, original_seq_len, hidden_size]，无效位置用 0 填充
        """
        if keep_indices is None:
            return flattened_hidden_states

        if keep_indices.ndim != 2:
            raise RuntimeError(
                f"keep_indices must be 2D [B, L_keep], got shape={tuple(keep_indices.shape)}"
            )

        batch_size, max_keep_len = keep_indices.shape
        hidden_size = flattened_hidden_states.shape[-1]
        device = flattened_hidden_states.device
        dtype = flattened_hidden_states.dtype

        # only keep legal destination positions; ignore invalid/out-of-range safely
        valid_mask = (keep_indices >= 0) & (keep_indices < original_seq_len)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(keep_indices)
        valid_batch_idx = batch_idx[valid_mask]
        valid_seq_idx = keep_indices[valid_mask].to(torch.int64)
        n_valid = int(valid_mask.sum().item())

        if flattened_hidden_states.ndim == 3:
            if flattened_hidden_states.shape[0] != batch_size:
                # Spec last-block path may return packed valid states as [1, N_valid, H].
                # Treat it as flattened valid tokens if token count matches n_valid.
                if flattened_hidden_states.shape[0] == 1 and batch_size > 1:
                    packed_valid = flattened_hidden_states.squeeze(0)
                    if packed_valid.shape[0] != n_valid:
                        raise RuntimeError(
                            "restore_hidden_states packed valid token count mismatch: "
                            f"hidden_states.shape={tuple(flattened_hidden_states.shape)}, "
                            f"expected_valid={n_valid}, "
                            f"keep_indices.shape={tuple(keep_indices.shape)}"
                        )
                    valid_hidden_states = packed_valid
                else:
                    raise RuntimeError(
                        "restore_hidden_states batch mismatch: "
                        f"hidden_states.shape={tuple(flattened_hidden_states.shape)}, "
                        f"keep_indices.shape={tuple(keep_indices.shape)}"
                    )
            else:
                # already full-length [B, original_seq_len, H], no restore needed
                if flattened_hidden_states.shape[1] == original_seq_len:
                    return flattened_hidden_states

                if flattened_hidden_states.shape[1] != max_keep_len:
                    raise RuntimeError(
                        "restore_hidden_states seq mismatch: "
                        f"hidden_states.shape={tuple(flattened_hidden_states.shape)}, "
                        f"keep_indices.shape={tuple(keep_indices.shape)}, "
                        f"original_seq_len={original_seq_len}"
                    )
                valid_hidden_states = flattened_hidden_states[valid_mask]

        elif flattened_hidden_states.ndim == 2:
            valid_hidden_states = flattened_hidden_states
            if valid_hidden_states.shape[0] != n_valid:
                raise RuntimeError(
                    "restore_hidden_states valid token count mismatch: "
                    f"hidden_states.shape={tuple(valid_hidden_states.shape)}, "
                    f"expected_valid={n_valid}, "
                    f"keep_indices.shape={tuple(keep_indices.shape)}"
                )
        else:
            raise RuntimeError(
                "restore_hidden_states expects hidden_states to be 2D or 3D, "
                f"got shape={tuple(flattened_hidden_states.shape)}"
            )

        restored_hidden_states = torch.zeros(
            batch_size, original_seq_len, hidden_size, dtype=dtype, device=device
        )
        restored_hidden_states[valid_batch_idx, valid_seq_idx, :] = valid_hidden_states
        return restored_hidden_states
    
    def _update_sequence(self, server_idx: int, block_idx: int, attempt_no: int) -> int:
        # If there is a failed server session, this code closes it
        self._exit_server_sessions(self._server_sessions[server_idx : server_idx + 1])

        n_prev_spans = len(self._server_sessions)
        update_end = self._server_sessions[server_idx].span.end if server_idx < n_prev_spans else self.num_blocks
        if attempt_no >= 1: 
            logger.debug(
                f"Due to a server failure, remote attention caches "
                f"from block {block_idx} to {update_end} will be regenerated"
            )

        updated_spans = self._sequence_manager.make_sequence(
            block_idx, update_end, mode="min_latency", cache_tokens_needed=self._max_length
        )

        # make_sequence() could return a longer sequence
        updated_spans[-1].end = min(updated_spans[-1].end, update_end)
        updated_sessions = self._enter_server_sessions(updated_spans)
        logger.debug(f"Found path from block {block_idx} to {update_end} via {len(updated_spans)} servers")
        
        
        # If there is a failed span, this code replaces it, otherwise it just adds new ones
        if server_idx < n_prev_spans:
            updated_sessions[0].history = self._server_sessions[server_idx].history
        self._server_sessions[server_idx : server_idx + 1] = updated_sessions

        # Update links to the next server session for direct server-to-server communication via rpc_push()
        for i in range(max(server_idx - 1, 0), min(server_idx + len(updated_spans), len(self._server_sessions) - 1)):
            self._server_sessions[i].next_session = self._server_sessions[i + 1]

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self._closed:
            self._exit_server_sessions(self._server_sessions)
            self._server_sessions.clear()
            self._closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()

    @property
    def last_token_id(self) -> Optional[torch.Tensor]:  # Backward compatibility with Petals < 2.1.0
        return self.output_ids[:, -1:] if self.output_ids is not None else None

    @last_token_id.setter
    def last_token_id(self, value: torch.Tensor):  # Backward compatibility with Petals < 2.1.0
        if self.output_ids is None:
            raise RuntimeError("Can't override `last_token_id` since the session has not stepped yet")
        self.output_ids[:, -1:] = value
