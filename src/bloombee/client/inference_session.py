from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import torch
from hivemind import MSGPackSerializer, anext, deserialize_torch_tensor, get_logger, serialize_torch_tensor
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.client.config import ClientConfig
from bloombee.client.routing import RemoteSequenceManager, maybe_log_traceback
from bloombee.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from bloombee.server.handler import TransformerConnectionHandler
from bloombee.utils.misc import DUMMY, DUMMY_INT64, is_dummy
from bloombee.utils.packaging import pack_args_kwargs

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
        assert start_from_position <= self._position
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

        n_input_tokens = inputs.shape[1] # get the number of token
        # print('client step() n_input_tokens', n_input_tokens)
        if self.history is None: # if the history log is empty
            self.history = inputs # assign the current inputs to the history log
        elif self.history.shape[1] == self._position: # if the length of the history equals the current position
            self.history = torch.cat([self.history, inputs[:, -n_input_tokens:]], dim=1) # 将当前输入的最后n_input_tokens个token拼接到历史记录中
        # assert self.history.shape[1] == self._position + n_input_tokens, ( #  确保历史记录的长度等于当前位置加上输入的token数量 
        #     f"Broken input cache: span={self.span} shape={self.history.shape} "
        #     f"position={self._position} n_input_tokens={n_input_tokens}"
        # )

        if not self.stepped: # if not exe step yet
            inputs = self.history  # Pass full inputs including prefix
        else:
            inputs = inputs[:, -n_input_tokens:]  # No need to pass prefix further

        # serialize inputs and put them into the queue
        # logger.info(f"kv_cache_position_ids: {kv_cache_position_ids}, draft_tokens: {draft_tokens}")
        combined_mask = self._pack_kv_into_tree_mask(tree_attention_mask, kv_cache_position_ids, draft_tokens)
        # logger.info(f"combined_mask: {combined_mask}")
        input_tensors, args_structure = pack_args_kwargs(inputs, prompts, hypo_ids, combined_mask)
        # logger.info(f"client inference session step() input_tensors after packing {input_tensors}")
        # logger.info(f"client inference session step() input_tensors after packing shape: {input_tensors[0].shape}")
        logger.info(f"_ServerInferenceSession  step id {step_id}")
        request_metadata = dict(session_id=self.session_id, step_id=step_id)
        if not self.stepped:
            request_metadata.update(self.session_metadata)
        if self._position is not None:
            request_metadata["start_from_position"] = self._position
        elif self.config.use_server_to_server:
            next_servers = self._collect_next_servers()
            if next_servers:
                request_metadata["next_servers"] = next_servers

        request_metadata["args_structure"] = args_structure

        # TODO: make possible to use different compression method for different tensors
        server_side_inference_schema, kwargs_schema = self.rpc_info["inference_schema"]
        compression = server_side_inference_schema[0].compression
        inference_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in input_tensors)

        # TODO: create more explicit way to check servers schema and client's structure
        # assert len(input_tensors) >= len(
        #     server_side_inference_schema
        # ), "Hidden_state, prompts and hypo_ids tensors are necessary for an inference step"

        # Serialize and send data (debug output removed for performance)
        serialized_tensors = [
            serialize_torch_tensor(tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(input_tensors, inference_schema)
        ]
        serialized_metadata = MSGPackSerializer.dumps(request_metadata)
        
        outputs_serialized = RemoteExpertWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=serialized_tensors,
                    metadata=serialized_metadata,
                )
            )
        )
        
        outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
        # assert (
        #     outputs[0].shape == inputs.shape
        # ), f"output activation shape is different from input shape: {outputs[0].shape} != {inputs.shape}"

        self._position += n_input_tokens
        logger.info(f"server inference session self._position: {self._position}")
        logger.info(f"server inference session output and shape: {len(outputs)},  {outputs} "  )
        return outputs
    
    def _pack_kv_into_tree_mask(self, tree_attention_mask, kv_cache_position_ids, draft_tokens):
        original_shape = tree_attention_mask.shape
        original_dtype = tree_attention_mask.dtype
        device = tree_attention_mask.device
        
        # logger.info(f"tree_attention_mask, shape: {tree_attention_mask.shape}, mask: {tree_attention_mask}")

        # shape (B, 1, seq_len)
        extra_row_shape = (original_shape[0], 1) + original_shape[2:]

        # ---- KV 行（倒数第6行） ----
        kv_extra_row = torch.full(extra_row_shape, 0, dtype=original_dtype, device=device)  # 改为0初始化
        if kv_cache_position_ids is not None:
            kv_flat = kv_cache_position_ids.flatten()
            available_space = kv_extra_row.numel() - 1
            actual_len = min(len(kv_flat), available_space)
            kv_extra_flat = kv_extra_row.view(-1)
            kv_extra_flat[:actual_len] = kv_flat[:actual_len].to(original_dtype)
            kv_extra_flat[-1] = actual_len  # 长度标记
        else:
            kv_extra_flat = kv_extra_row.view(-1)
            kv_extra_flat[-1] = 255  # 无数据标记

        # ---- Draft 行（最后5行） ----
        draft_rows = []
        logger.info(f"_pack_kv_into_tree_mask, shape: {draft_tokens.shape} draft_tokens: {draft_tokens}")
        if draft_tokens is not None:
            draft_flat = draft_tokens.flatten()
            payload_per_row = kv_extra_row.numel() - 1  # 每行可写的数据数量（留一个位置给长度标记）
            
            for i in range(2):  # 5行
                row = torch.zeros(extra_row_shape, dtype=original_dtype, device=device)  # 初始化为0
                row_flat = row.view(-1)
                start = i * payload_per_row
                
                if start < len(draft_flat):
                    actual_len = min(len(draft_flat) - start, payload_per_row)
                    # 写入实际的draft tokens
                    row_flat[:actual_len] = draft_flat[start:start+actual_len].to(original_dtype)
                    row_flat[-1] = actual_len  # 最后一个元素存储长度
                else:
                    # 无数据的行
                    row_flat[-1] = 255
                    
                draft_rows.append(row)
        else:
            # 没有draft tokens，创建5个空行
            for _ in range(2):
                row = torch.zeros(extra_row_shape, dtype=original_dtype, device=device)
                row.view(-1)[-1] = 255  # 标记为无数据
                draft_rows.append(row)

        # ---- 拼接 ----
        combined_mask = torch.cat([tree_attention_mask, kv_extra_row] + draft_rows, dim=1)
        return combined_mask

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
        step_id = str(uuid.uuid4()) #生成一个唯一的步骤 ID。

        n_input_tokens = inputs.shape[1]
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {self._position} + current {n_input_tokens} exceeds pre-allocated maximum {self._max_length}"
            )

        server_idx = 0
        block_idx = 0
        
        logger.info(f"hidden states in inference session: {inputs.shape}, {inputs}")
        
        while block_idx < self.num_blocks:
            for attempt_no in itertools.count():
                logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
                server_session = None
                try:
                    if not self._server_sessions or attempt_no >= 1:
                        self._update_sequence(server_idx, block_idx, attempt_no)

                    server_session = self._server_sessions[server_idx]
                    assert server_session.position == self.position, f"{server_session.position} and {self.position}"

                    inputs, keep_indices = server_session.step(
                        inputs,
                        prompts[server_session.span.start : server_session.span.end],
                        hypo_ids,
                        tree_attention_mask,
                        kv_cache_position_ids,
                        draft_tokens,
                        step_id=step_id,
                    )
                    
                    logger.info(f"get output and keep_indices: {keep_indices}, kv_cache_position_ids: {kv_cache_position_ids}")
                    kv_cache_position_ids = self._update_kv_cache_position_ids(kv_cache_position_ids, self.keep_indices)
                    self.keep_indices = keep_indices
                    prefill_length = n_input_tokens - tree_attention_mask.shape[1]
                    attention_mask_indices = keep_indices[prefill_length:] - prefill_length
                    # logger.info(f"attention_mask_indices: {attention_mask_indices}, tree_attention_mask: {tree_attention_mask}")
                    tree_attention_mask = tree_attention_mask[:, attention_mask_indices, attention_mask_indices]
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
        print(f"lient inference session outputs before recover, inputs: {inputs.shape}, {inputs}")
        if draft_tokens is not None:
            inputs = self._recover_hidden_states(inputs, self.keep_indices, draft_tokens.shape[1])
            
        logger.info(f"hidden states in inference session after inference: {inputs.shape}, {inputs}")
        outputs = inputs 
        
        outputs = outputs.to(device=inputs_device, dtype=inputs_dtype) 
        # print('client inference session outputs ', outputs.shape)
        return outputs
    
    def _recover_hidden_states(self, hidden_states, keep_indices, original_length):
        if not torch.is_tensor(keep_indices):
            keep_indices = torch.tensor(keep_indices, device=hidden_states.device, dtype=torch.long)

        # 判断是否有 batch 维度
        if hidden_states.dim() == 2:
            # [S_kept, H]
            recovered = hidden_states.new_zeros((original_length, hidden_states.size(-1)))
            recovered[keep_indices] = hidden_states

        elif hidden_states.dim() == 3:
            # [B, S_kept, H]
            B, _, H = hidden_states.shape
            recovered = hidden_states.new_zeros((B, original_length, H))
            recovered[:, keep_indices, :] = hidden_states

        else:
            raise ValueError(f"Unexpected hidden_states dim {hidden_states.dim()}, expected 2 or 3")

        # 保存 mask 方便后续屏蔽 logits
        mask = torch.ones(original_length, dtype=torch.bool, device=hidden_states.device)
        mask[keep_indices] = False
        self._last_padded_mask = mask

        return recovered
    
    def _update_kv_cache_position_ids(self, kv_cache_position_ids, keep_indices):
        if kv_cache_position_ids is None:
            return
        
        if not torch.is_tensor(keep_indices):
            keep_indices = torch.tensor(keep_indices, device=kv_cache_position_ids.device)

        # 建立映射表：原始 id → 新位置
        # 例如 keep_indices = [2,4,7] → map: {2:0, 4:1, 7:2}
        mapping = {int(k.item()): i for i, k in enumerate(keep_indices)}

        # 生成新 ids：查表
        new_ids = torch.tensor(
            [mapping.get(int(x.item()), -1) for x in kv_cache_position_ids],
            device=kv_cache_position_ids.device
        )

        return new_ids

    
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