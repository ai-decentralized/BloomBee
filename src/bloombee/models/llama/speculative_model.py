from typing import Optional, Union, List, Tuple, Any

import copy
import math
import torch
import time
import numpy as np
import contextlib
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from bloombee.models.llama.config import DistributedLlamaConfig
from bloombee.models.llama.model import DistributedLlamaForCausalLM
from bloombee.models.llama.spec_decoding_drafter import MultiSSMDrafter
from bloombee.models.llama.spec_decoding_verify import verify_path
from bloombee.models.llama.spe_dec_tree import SpeculativeTree, TreeNode, prepare_incremental_tree_batch

from bloombee.client.remote_generation import RemotePastKeyValues
from bloombee.client.inference_session import InferenceSession
from hivemind.utils.logging import get_logger

logger = get_logger()

_GENERATION_CONFIG_KWARGS = (
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "pad_token_id",
    "eos_token_id",
    "bos_token_id",
)


def _eos_token_ids(generation_config: GenerationConfig) -> Tuple[int, ...]:
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        return ()
    if isinstance(eos, int):
        return (int(eos),)
    if isinstance(eos, torch.Tensor):
        return tuple(int(x) for x in eos.detach().cpu().view(-1).tolist())
    if isinstance(eos, (list, tuple, set)):
        return tuple(int(x) for x in eos)
    return (int(eos),)


def _cap_valid_lengths_to_remaining(
    valid_lengths: torch.LongTensor,
    seq_lengths: torch.LongTensor,
    initial_len: int,
    max_new_tokens: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    remaining = torch.clamp(
        int(max_new_tokens) - (seq_lengths - int(initial_len)),
        min=0,
    )
    capped_valid_lengths = torch.minimum(valid_lengths, remaining)
    append_llm_token = (remaining > valid_lengths).to(dtype=torch.long)
    return capped_valid_lengths, append_llm_token


def _merge_generation_config_kwargs(
    generation_config: GenerationConfig,
    model_kwargs: dict,
) -> GenerationConfig:
    """Honor common ``generate(..., do_sample=False)`` kwargs.

    HF's ``GenerationMixin.generate`` folds these kwargs into a copied
    GenerationConfig before decoding. BloomBee's speculative path bypasses that
    helper, so do the small compatible subset explicitly.
    """
    merged = copy.deepcopy(generation_config)
    for key in _GENERATION_CONFIG_KWARGS:
        if key in model_kwargs:
            value = model_kwargs.pop(key)
            if value is not None:
                setattr(merged, key, value)
    return merged


class DistributedLlamaForSpeculativeGeneration(DistributedLlamaForCausalLM):
    def __init__(self, config: DistributedLlamaConfig):
        super().__init__(config)
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        drafter: MultiSSMDrafter,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        streamer: Optional["BaseStreamer"] = None,
        beam_width: Union[int, List[int]] = 1,
        max_tree_depth: int = 4,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 128,
        session_max_length: Optional[int] = None,
        tree_budget: Optional[int] = None,
        tree_min_log_prob: Optional[float] = None,
        **model_kwargs,
    ) -> torch.LongTensor:

        generation_config = generation_config or getattr(self, "generation_config", GenerationConfig())
        generation_config = _merge_generation_config_kwargs(generation_config, model_kwargs)
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        # Do not override do_sample here. When do_sample=False the verify loop
        # takes the argmax path (token-identical to greedy decoding on the
        # target model, same as before). When do_sample=True the verify loop
        # uses SpecInfer rejection sampling (arXiv 2305.09781) which is
        # provably distribution-equivalent to sampling directly from the
        # target model.
        generation_config.return_dict_in_generate = False

        # Resolve session_max_length from (in order): kwarg > model_kwargs > config
        # > a conservative fallback. The previous hardcoded 624 made every
        # speculation session request a 624-token cache regardless of actual
        # prompt + max_new_tokens, which wastes KV budget and constrains
        # admission under continuous batching. We use ceil(prompt + budget)
        # with a floor that matches drafter tree growth.
        kwarg_override = session_max_length
        session_max_length = model_kwargs.pop("session_max_length", kwarg_override)
        effective_tree_budget = tree_budget
        if effective_tree_budget is None:
            effective_tree_budget = getattr(drafter, "default_tree_budget", None)
        if session_max_length is None:
            prompt_len = int(input_ids.shape[1])
            # Sequoia plan: total tree budget is sum of products of widths.
            if isinstance(beam_width, (list, tuple)):
                tree_nodes = 0
                running = 1
                for w in beam_width:
                    running *= max(int(w), 1)
                    tree_nodes += running
            else:
                tree_nodes = max_tree_depth * max(int(beam_width), 1)
            if effective_tree_budget is not None:
                tree_nodes = max(tree_nodes, int(effective_tree_budget))
            # Each decode step pushes the full tree into the server cache
            # (only the verified prefix is retained in the rollback below, but
            # the server must have room to store the candidates first). Size
            # for max_new_tokens worth of steps with one verified token each,
            # plus one full tree-sized spike per step.
            session_max_length = max(
                prompt_len + int(max_new_tokens) * (tree_nodes + 1) + 32,
                prompt_len + 256,
            )
        logger.info(
            "Speculative session_max_length=%s (prompt=%s max_new_tokens=%s depth=%s width=%s)",
            session_max_length,
            int(input_ids.shape[1]),
            max_new_tokens,
            max_tree_depth,
            beam_width,
        )

        # Use inference session for proper distributed caching
        with self.transformer.h.inference_session(max_length=session_max_length) as session:
            return self._sample_with_session(
                input_ids=input_ids,
                drafter=drafter,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                session=session,
                streamer=streamer,
                beam_width=beam_width,
                max_tree_depth=max_tree_depth,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window,
                max_new_tokens=max_new_tokens,
                tree_budget=effective_tree_budget,
                tree_min_log_prob=tree_min_log_prob,
                **model_kwargs,
            )

    def _sample_with_session(
        self,
        input_ids: torch.LongTensor,
        drafter: MultiSSMDrafter,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        session: InferenceSession,
        streamer: Optional["BaseStreamer"],
        beam_width: Union[int, List[int]] = 2,
        max_tree_depth: int = 3,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 128,
        tree_budget: Optional[int] = None,
        tree_min_log_prob: Optional[float] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        logger.info("Starting speculative decoding with distributed inference session!")
        eos_token_ids = _eos_token_ids(generation_config)
        eos_token_tensor = (
            torch.tensor(eos_token_ids, dtype=torch.long, device=input_ids.device)
            if eos_token_ids
            else None
        )
        has_eos_stopping_criteria = bool(eos_token_ids) or any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        finished = False
        
        # Initialize past_key_values for session tracking
        past_key_values = RemotePastKeyValues()
        batch_positions = torch.full(
            (batch_size,), 
            session.position,
            dtype=torch.long,
            device="cuda"
        )
        past_key_values.update_seen(batch_positions)
        past_key_values.set_is_spec_decoding(torch.tensor([1], dtype=torch.long, device="cuda"))
        
        is_first_iteration = True
        step_idx = 0
        current_input_ids = input_ids
        llm_generated_token = None
        # EAGLE drafter conditioning state — populated after the first verify.
        # None on the very first iteration; SSM drafters ignore both kwargs.
        prev_last_hidden: Optional[torch.Tensor] = None
        prev_last_token: Optional[torch.Tensor] = None
        # Target hidden states for committed tokens before the current root.
        # EAGLE's official drafter consumes the whole shifted prefix, not just
        # the endpoint hidden. We maintain a padded [B, L, H] buffer where
        # L == seq_lengths[b] - 1 for each active row.
        eagle_prefix_hidden_states: Optional[torch.Tensor] = None

        # 新增：维护每个序列的真实长度
        seq_lengths = torch.full((batch_size,), input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        ignore_token_ids: list = [0, 2]
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in ignore_token_ids:
            valid_mask = valid_mask & (input_ids != token_id)
        
        # 计算每个序列的有效 token 数量
        seq_lengths = valid_mask.sum(dim=1)  # [batch_size]
        past_key_values.set_prefill_length(seq_lengths)
        
        pad_token_id = generation_config.pad_token_id if generation_config.pad_token_id is not None else 0
        logger.info(f"init input_ids: {input_ids}, seq_lengths: {seq_lengths}")
        # 修改循环条件：基于最短序列的长度判断
        # t0 = time.perf_counter()
        initial_len = input_ids.shape[1]
        t0 = time.perf_counter()  # 用于记录第一个达标的时间
        has_printed_first_reach = False # 确保只打印一次
        sample_finish_times = [None] * batch_size
        sample_finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        while not finished and (seq_lengths.min().item() - initial_len) < max_new_tokens:
            # 1. Build speculative trees using SSM - 传入 seq_lengths
            t1 = time.perf_counter()
            # Pass EAGLE-2-style tree-budget kwargs through to the drafter.
            # MultiSSMDrafter consumes them to do post-expansion budget pruning;
            # EAGLEDrafter uses them to drive its dynamic tree-growth loop.
            # ``prev_last_hidden`` / ``prev_last_token`` give the drafter the
            # target's last committed hidden + token id (EAGLE conditioning).
            drafter_kwargs = {
                'tree_budget': tree_budget,
                'tree_min_log_prob': tree_min_log_prob,
            }
            if prev_last_hidden is not None:
                drafter_kwargs['prev_last_hidden'] = prev_last_hidden
                drafter_kwargs['prev_last_token'] = prev_last_token
            if eagle_prefix_hidden_states is not None:
                drafter_kwargs['prefix_hidden_states'] = eagle_prefix_hidden_states
            try:
                spec_trees = drafter.build_trees_parallel(
                    current_input_ids, seq_lengths, beam_width, max_tree_depth,
                    **{k: v for k, v in drafter_kwargs.items() if v is not None},
                )
            except TypeError:
                # Older drafters don't take EAGLE kwargs; retry plain.
                spec_trees = drafter.build_trees_parallel(
                    current_input_ids, seq_lengths, beam_width, max_tree_depth,
                )
            t2 = time.perf_counter()
            logger.info(f"Step {step_idx}: Built speculative trees in {t2 - t1:.4f} seconds")
            # logger.info(f"spec_trees, {spec_trees}")
            
            # 2. Verify trees using distributed inference
            (
                verified_tokens,
                verified_tokens_positions,
                past_key_values,
                llm_generated_token,
                valid_lengths,
                verify_hidden_states,
                final_positions,
            ) = self._verify_trees_with_forward(
                input_ids=current_input_ids,
                llm_generated_token=llm_generated_token,
                trees=spec_trees,
                logits_processor=logits_processor,
                past_key_values=past_key_values,
                is_first_iteration=is_first_iteration,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window,
                seq_lengths=seq_lengths,
                do_sample=bool(getattr(generation_config, "do_sample", False)),
                temperature=float(getattr(generation_config, "temperature", 1.0) or 1.0),
            )

            old_seq_lengths = seq_lengths.clone()
            valid_lengths, append_llm_token = _cap_valid_lengths_to_remaining(
                valid_lengths=valid_lengths,
                seq_lengths=seq_lengths,
                initial_len=initial_len,
                max_new_tokens=max_new_tokens,
            )
            if verified_tokens_positions is not None:
                position_offsets = torch.arange(
                    verified_tokens_positions.shape[1],
                    device=verified_tokens_positions.device,
                )
                keep_positions = position_offsets.unsqueeze(0) <= valid_lengths.unsqueeze(1)
                verified_tokens_positions = verified_tokens_positions.masked_fill(
                    ~keep_positions,
                    -1,
                )
            if verified_tokens is not None:
                max_valid_length = int(valid_lengths.max().item())
                if max_valid_length == 0:
                    verified_tokens = None
                elif verified_tokens.shape[1] > max_valid_length:
                    verified_tokens = verified_tokens[:, :max_valid_length]

            # M1/M3 plumbing: gather the committed-endpoint hidden state and
            # maintain the full committed-prefix hidden buffer for EAGLE.
            if verify_hidden_states is not None and final_positions is not None:
                idx = torch.arange(verify_hidden_states.size(0), device=verify_hidden_states.device)
                prev_last_hidden = verify_hidden_states[idx, final_positions, :].detach()
                prev_last_token = llm_generated_token[idx, 0].detach() if llm_generated_token is not None else None
                if getattr(drafter, "uses_eagle_hidden_states", False):
                    eagle_prefix_hidden_states = self._update_eagle_prefix_hidden_states(
                        prefix_hidden_states=eagle_prefix_hidden_states,
                        verify_hidden_states=verify_hidden_states.detach(),
                        kv_cache_position_ids=verified_tokens_positions,
                        old_seq_lengths=old_seq_lengths,
                        is_first_iteration=is_first_iteration,
                    )
            else:
                prev_last_hidden = None
                prev_last_token = None
            
            t3 = time.perf_counter()
            logger.info(f"Step {step_idx}: Verified trees with distributed inference in {t3 - t2:.4f} seconds")
            
            # logger.info(f"verified_tokens_positions: {verified_tokens_positions}")
            
            past_key_values.set_kv_cache(verified_tokens_positions)
            
            is_first_iteration = False
            
            # 3. Apply stopping conditions
            if has_eos_stopping_criteria:
                if verified_tokens is not None:
                    verified_tokens = verified_tokens * unfinished_sequences.unsqueeze(-1) + pad_token_id * (
                        1 - unfinished_sequences.unsqueeze(-1)
                    )
                llm_generated_token = llm_generated_token * unfinished_sequences.unsqueeze(-1) + pad_token_id * (
                    1 - unfinished_sequences.unsqueeze(-1)
                )

            # 4. Update input sequence with proper padding handling
            # logger.info(f"current_input_ids: {current_input_ids}")
            # logger.info(f"verified_tokens: {verified_tokens}")
            # logger.info(f"llm_generated_token: {llm_generated_token}")
            # logger.info(f"valid_lengths: {valid_lengths}")
            # logger.info(f"seq_lengths: {seq_lengths}")
            current_input_ids, seq_lengths = self._update_input_ids_with_padding(
                current_input_ids=current_input_ids,
                verified_tokens=verified_tokens,
                llm_generated_token=llm_generated_token,
                valid_lengths=valid_lengths,
                seq_lengths=seq_lengths,
                pad_token_id=pad_token_id,
                append_llm_token=append_llm_token,
            )

            if eos_token_tensor is not None:
                for i in range(batch_size):
                    start = int(old_seq_lengths[i].item())
                    end = int(seq_lengths[i].item())
                    if end > start:
                        new_tokens = current_input_ids[i, start:end]
                        if torch.isin(new_tokens, eos_token_tensor).any():
                            unfinished_sequences[i] = 0
            
            # t4 = time.perf_counter()
            # logger.info(f"Step {step_idx}: Updated input_ids with padding in {t4 - t3:.4f} seconds")
            
            # logger.info(f"current_input_ids: {current_input_ids}, seq_lengths: {seq_lengths}")

            if streamer is not None:
                # Stream 时根据 valid_lengths 只输出有效 token
                for i in range(batch_size):
                    if unfinished_sequences[i]:
                        if verified_tokens is not None and valid_lengths[i] > 0:
                            streamer.put(verified_tokens[i, :valid_lengths[i]].cpu())
                        if append_llm_token[i]:
                            streamer.put(llm_generated_token[i].cpu())

            # 5. Check if finished
            unfinished_sequences = unfinished_sequences & (
                (seq_lengths - initial_len) < max_new_tokens
            ).long()
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(current_input_ids, None)
            finished = unfinished_sequences.max() == 0
            total_time = time.perf_counter() - t1
            logger.info(f"Step {step_idx}: FTotal Time Elapsed={total_time:.4f} seconds")
            current_generations = seq_lengths - initial_len
            for i in range(batch_size):
                if (current_generations[i] >= max_new_tokens and not sample_finished[i]):
                    finish_time = time.perf_counter() - t0
                    sample_finish_times[i] = finish_time
                    sample_finished[i] = True
                    logger.info(f"step {step_idx} Sample {i} finished generation ({max_new_tokens} tokens) at {finish_time:.4f}s")
            step_idx += 1

        if streamer is not None:
            streamer.end()
            
        logger.info("====== Batch Generation Summary ======")
        for i, t in enumerate(sample_finish_times):
            if t is not None:
                logger.info(f"Sample {i}: finished at {t:.4f}s")
            else:
                logger.info(f"Sample {i}: did not reach max_new_tokens")
        
        return current_input_ids

    def _update_eagle_prefix_hidden_states(
        self,
        *,
        prefix_hidden_states: Optional[torch.Tensor],
        verify_hidden_states: torch.Tensor,
        kv_cache_position_ids: torch.Tensor,
        old_seq_lengths: torch.LongTensor,
        is_first_iteration: bool,
    ) -> torch.Tensor:
        """Append target hiddens that are committed before the next root token.

        After a speculative verify step, the next root is the bonus token
        sampled from the target logits. EAGLE should be conditioned on target
        hidden states for every token *before* that bonus. For a normal tree
        verify those hiddens are root + accepted draft tokens; for the initial
        no-tree warmup they are the full prompt hiddens.
        """
        batch_size = verify_hidden_states.shape[0]
        hidden_size = verify_hidden_states.shape[-1]
        device = verify_hidden_states.device
        dtype = verify_hidden_states.dtype
        rows: List[torch.Tensor] = []

        for b in range(batch_size):
            old_len = int(old_seq_lengths[b].item())
            if is_first_iteration and prefix_hidden_states is None:
                # Warmup/prefill fallback: verify_hidden_states is aligned with
                # the full prompt, so keep all prompt hiddens before the sampled
                # bonus root.
                rows.append(verify_hidden_states[b, :old_len, :])
                continue

            previous = (
                prefix_hidden_states[b, : max(old_len - 1, 0), :].to(device=device, dtype=dtype)
                if prefix_hidden_states is not None
                else verify_hidden_states.new_empty((0, hidden_size))
            )
            root_abs = old_len - 1
            selected = kv_cache_position_ids[b]
            selected = selected[selected >= 0]
            if selected.numel() == 0:
                rows.append(previous)
                continue
            rel = (selected.to(device=device) - root_abs).long()
            rel = rel[(rel >= 0) & (rel < verify_hidden_states.shape[1])]
            if rel.numel() == 0:
                rows.append(previous)
            else:
                rows.append(torch.cat([previous, verify_hidden_states[b, rel, :]], dim=0))

        max_len = max((row.shape[0] for row in rows), default=0)
        out = verify_hidden_states.new_zeros((batch_size, max_len, hidden_size))
        for b, row in enumerate(rows):
            if row.numel() > 0:
                out[b, :row.shape[0], :] = row
        return out

    def _update_input_ids_with_padding(
        self,
        current_input_ids: torch.LongTensor,
        verified_tokens: Optional[torch.LongTensor],
        llm_generated_token: torch.LongTensor,
        valid_lengths: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        pad_token_id: int,
        append_llm_token: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        更新 input_ids，处理不同序列验证通过的 token 数量不同的情况
        
        Returns:
            updated_input_ids: 更新后的 input_ids，padding 对齐
            updated_seq_lengths: 更新后的每个序列真实长度
        """
        batch_size = current_input_ids.shape[0]
        device = current_input_ids.device

        if append_llm_token is None:
            append_llm_token = torch.ones_like(valid_lengths, dtype=torch.long)
        else:
            append_llm_token = append_llm_token.to(device=device, dtype=torch.long)
        
        # 计算每个序列需要添加的 token 数（verified + optional llm token）
        tokens_to_add = valid_lengths + append_llm_token  # [batch_size]
        
        # 计算新的序列长度
        new_seq_lengths = seq_lengths + tokens_to_add
        new_max_len = new_seq_lengths.max().item()
        
        # 创建新的 input_ids tensor
        new_input_ids = torch.full(
            (batch_size, new_max_len), 
            pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        for i in range(batch_size):
            old_len = seq_lengths[i].item()
            
            # 复制原有的有效 token
            new_input_ids[i, :old_len] = current_input_ids[i, :old_len]
            
            # 添加验证通过的 token
            v_len = valid_lengths[i].item()
            if v_len > 0 and verified_tokens is not None:
                new_input_ids[i, old_len:old_len + v_len] = verified_tokens[i, :v_len]

            if append_llm_token[i].item():
                new_input_ids[i, old_len + v_len] = llm_generated_token[i, 0]
        
        return new_input_ids, new_seq_lengths
    
    def _verify_trees_with_forward(
        self,
        input_ids: torch.LongTensor,
        llm_generated_token: torch.Tensor,
        trees: List[SpeculativeTree],
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        is_first_iteration: bool,
        use_kv_cache: bool,
        kv_cache_window: int,
        seq_lengths: torch.LongTensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[
        Optional[torch.LongTensor],
        torch.Tensor,
        RemotePastKeyValues,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.LongTensor],
    ]:
        """
        Verify speculative trees using standard forward() call within the active session context
        
        Returns:
            verified_tokens: [batch_size, max_verified_len] 或 None
            kv_cache_position_ids: [batch_size, max_pos_len]
            past_key_values: 更新后的 past_key_values
            llm_generated_tokens: [batch_size, 1]
            valid_lengths: [batch_size] 每个序列验证通过的 token 数
        """
        # logger.info(f"input_ids: {input_ids}")
        # logger.info(f"seq_lengths: {seq_lengths}")
        # logger.info(f"kv_cache_position_ids: {past_key_values.kv_cache_position_ids}")
        tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
            trees, input_ids, input_ids.device, seq_lengths=seq_lengths, is_prefill=is_first_iteration, kv_cache_position_ids=past_key_values.kv_cache_position_ids
        )
        
        # logger.info(f"tree_tokens: {tree_tokens}, attention_mask: {attention_mask.shape}")
        # logger.info(f"attention_mask: {attention_mask}")
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if attention_mask is None or tree_tokens.shape[1] == 0:
            logger.warning("No tree tokens to verify, falling back to regular generation")
            (
                fallback_token,
                fallback_positions,
                fallback_hidden_states,
                fallback_final_positions,
            ) = self._fallback_generation_with_forward(
                input_ids,
                logits_processor,
                past_key_values,
                seq_lengths,
                is_first_iteration=is_first_iteration,
                do_sample=do_sample,
                temperature=temperature,
            )
            valid_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            return (
                None,
                fallback_positions,
                past_key_values,
                fallback_token,
                valid_lengths,
                fallback_hidden_states,
                fallback_final_positions,
            )
        
        # tree_mask_packed = self.pack_bool_mask_to_int64(attention_mask)
        tree_mask_packed = attention_mask
        # logger.info(f"tree_mask_packed: {tree_mask_packed}")
        
        with torch.no_grad():
            if not use_kv_cache:
                # No cache: process tree tokens directly
                logger.warning("Processing without KV cache, may cause error!!!")
                # Split forward so verify can hold both hidden states and
                # logits separately. EAGLE-2 needs h_last per iteration; the
                # naive HF path discards the hidden after lm_head().
                model_out = self.model(
                    input_ids=tree_tokens,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=False,
                )
                hidden_states = model_out.last_hidden_state
                logits = self.lm_head(hidden_states)
                new_past_key_values = past_key_values
                
            elif is_first_iteration or past_key_values is None:
                # First iteration: process full sequence to establish cache
                # 需要根据 seq_lengths 构建正确的 full_sequence
                max_seq_len = seq_lengths.max().item()
                full_sequence = torch.cat([input_ids[:, :max_seq_len], tree_tokens], dim=-1)
                
                model_out = self.model(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = model_out.last_hidden_state
                logits = self.lm_head(hidden_states)

                if past_key_values is None:
                    new_past_key_values = RemotePastKeyValues()
                else:
                    new_past_key_values = past_key_values
                
            else:
                # Subsequent iterations: use existing cache
                active_session = self.transformer.h.active_session
                if active_session is None:
                    raise ValueError("No active session available for cached inference")
                
                # Handle cache window management
                if active_session.position > kv_cache_window:
                    trim_amount = active_session.position - kv_cache_window
                    active_session.position = kv_cache_window
                    
                if llm_generated_token is None:
                    full_sequence = tree_tokens
                else:
                    full_sequence = torch.cat([llm_generated_token, tree_tokens], dim=-1)
                
                model_out = self.model(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = model_out.last_hidden_state
                logits = self.lm_head(hidden_states)
                new_past_key_values = past_key_values
                new_past_key_values.update_seen(active_session.position)
                
        # Extract verification results — also returns final_positions [B] so
        # callers can gather the committed-endpoint hidden state for EAGLE.
        (
            verified_tokens,
            kv_cache_position_ids,
            llm_generated_tokens,
            valid_lengths,
            final_positions,
        ) = self._extract_best_verified_paths_fixed(
            logits, batch_node_paths, input_ids, logits_processor, tree_tokens.shape[1], seq_lengths, is_first_iteration,
            do_sample=do_sample, temperature=temperature,
        )
        return (
            verified_tokens,
            kv_cache_position_ids,
            new_past_key_values,
            llm_generated_tokens,
            valid_lengths,
            hidden_states,
            final_positions,
        )
    
    def pack_bool_mask_to_int64(self, mask_bool: torch.Tensor) -> torch.Tensor:
        assert mask_bool.dtype == torch.bool, "Input must be a bool tensor"
        return mask_bool.to(dtype=torch.int64)
    
    def _fallback_generation_with_forward(
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        seq_lengths: torch.LongTensor,
        *,
        is_first_iteration: bool,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.LongTensor]:
        """
        Fallback to regular generation using forward() call within active session
        """
        def _pick_next(processed_logits: torch.Tensor) -> torch.LongTensor:
            if do_sample:
                temp = float(temperature) if temperature and temperature > 0 else 1.0
                probs = torch.softmax(processed_logits / temp, dim=-1)
                return torch.multinomial(probs, 1)
            return torch.argmax(processed_logits, dim=-1, keepdim=True)

        try:
            logger.info("[DEBUG] Using fallback generation")
            
            batch_size = input_ids.shape[0]
            device = input_ids.device
            old_spec_flag = past_key_values.is_spec_decoding
            # This is a regular target-model step, not tree verification. Run
            # it with the compact non-spec path so the server cache is seeded
            # with the true prefix/root instead of a degenerate tree layout.
            past_key_values.set_is_spec_decoding(torch.tensor([0], dtype=torch.long, device=device))
            
            if is_first_iteration:
                max_seq_len = int(seq_lengths.max().item())
                model_out = self.model(
                    input_ids=input_ids[:, :max_seq_len],
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = model_out.last_hidden_state
                logits_full = self.lm_head(hidden_states)
                logits = torch.stack(
                    [logits_full[i, int(seq_lengths[i].item()) - 1, :] for i in range(batch_size)],
                    dim=0,
                )
                final_positions = (seq_lengths - 1).to(device=device)
                kv_cache_position_ids = final_positions[:, None]
            else:
                # 获取每个序列最后一个有效 token; this root token has not yet
                # been run through the target, so use the existing cache plus
                # one regular decode step and sample the bonus from its logits.
                last_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                for i in range(batch_size):
                    last_pos = seq_lengths[i].item() - 1
                    last_tokens[i, 0] = input_ids[i, last_pos]

                model_out = self.model(
                    input_ids=last_tokens,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = model_out.last_hidden_state
                logits = self.lm_head(hidden_states)[:, -1, :]
                final_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
                kv_cache_position_ids = (seq_lengths - 1).to(device=device)[:, None]
            
            # Apply logits processors
            processed_logits = logits
            for processor in logits_processor:
                processed_logits = processor(input_ids, processed_logits)
            
            next_token = _pick_next(processed_logits)
            past_key_values.set_is_spec_decoding(old_spec_flag)

            return next_token, kv_cache_position_ids, hidden_states, final_positions
            
        except Exception as e:
            if "old_spec_flag" in locals():
                try:
                    past_key_values.set_is_spec_decoding(old_spec_flag)
                except Exception:
                    pass
            logger.error(f"Fallback generation failed: {e}")
            eos_token_id = getattr(self.config, 'eos_token_id', 2)
            batch_size = input_ids.shape[0]
            device = input_ids.device
            empty_hidden = torch.zeros(
                batch_size,
                1,
                self.config.hidden_size,
                dtype=torch.float32,
                device=device,
            )
            return (
                torch.full((batch_size, 1), eos_token_id, device=device),
                (seq_lengths - 1).to(device=device)[:, None],
                empty_hidden,
                torch.zeros(batch_size, dtype=torch.long, device=device),
            )
    
    def _build_speculative_trees_batched(
        self, 
        input_ids: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int, 
        max_depth: int,
        seq_lengths: torch.LongTensor,
    ) -> List[SpeculativeTree]:
        """Build speculative trees - 所有样本按 depth 批量处理"""
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        pad_token_id = getattr(ssm.config, 'pad_token_id', 0)
        
        # 初始化所有 trees
        trees = []
        valid_inputs = []  # 每个样本的有效 input_ids
        
        for batch_idx in range(batch_size):
            actual_len = seq_lengths[batch_idx].item()
            valid_input_ids = input_ids[batch_idx, :actual_len]
            valid_inputs.append(valid_input_ids)
            
            root_token = valid_input_ids[-1].item()
            tree = SpeculativeTree(root_token, f"req_{batch_idx}")
            trees.append(tree)
        
        # 按 depth 循环，每层一次 SSM 调用
        for depth in range(max_depth):
            
            # 收集所有样本在当前 depth 的 nodes 和 contexts
            all_contexts = []
            node_mapping = []  # (batch_idx, node) 用于结果拆分
            
            for batch_idx in range(batch_size):
                tree = trees[batch_idx]
                valid_input_ids = valid_inputs[batch_idx]
                root_token = valid_input_ids[-1].item()
                
                current_nodes = tree.get_nodes_at_depth(depth)
                
                for node in current_nodes:
                    path_to_node = node.get_path_from_root()
                    context = torch.cat([
                        valid_input_ids[:-1],
                        torch.tensor([root_token] + path_to_node, device=device)
                    ])
                    all_contexts.append(context)
                    node_mapping.append((batch_idx, node))
            
            # 如果没有 context，结束
            if not all_contexts:
                break
            
            # Padding 成统一长度
            max_len = max(len(ctx) for ctx in all_contexts)
            padded_contexts = []
            attention_masks = []
            
            for ctx in all_contexts:
                pad_len = max_len - len(ctx)
                padded = torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=torch.long, device=device),
                    ctx
                ])
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=device),
                    torch.ones(len(ctx), dtype=torch.long, device=device)
                ])
                padded_contexts.append(padded)
                attention_masks.append(mask)
            
            batch_contexts = torch.stack(padded_contexts)
            batch_masks = torch.stack(attention_masks)
            
            # 一次 SSM forward 处理所有
            with torch.no_grad():
                outputs = ssm(batch_contexts, attention_mask=batch_masks, use_cache=False)
                all_logits = outputs.logits[:, -1, :]  # (total_nodes, vocab_size)
            
            # 按样本分组处理结果
            # 先按 batch_idx 分组
            batch_node_results = {}  # batch_idx -> [(node, candidates), ...]
            
            for i, (batch_idx, node) in enumerate(node_mapping):
                logits = all_logits[i]
                _, top_k_indices = torch.topk(logits, k=beam_width)
                probs = torch.softmax(logits, dim=-1)
                
                candidates = []
                for j in range(beam_width):
                    token_id = top_k_indices[j].item()
                    prob = probs[token_id].item()
                    candidates.append((token_id, prob))
                
                if batch_idx not in batch_node_results:
                    batch_node_results[batch_idx] = []
                batch_node_results[batch_idx].append((node, candidates))
            
            # 更新每个 tree
            any_new_nodes = False
            for batch_idx in range(batch_size):
                if batch_idx not in batch_node_results:
                    continue
                
                tree = trees[batch_idx]
                node_candidates = batch_node_results[batch_idx]
                
                # 保持顺序：nodes 和 candidates_per_node 要对应
                nodes = [nc[0] for nc in node_candidates]
                candidates_per_node = [nc[1] for nc in node_candidates]
                
                try:
                    new_nodes = tree.add_layer(nodes, candidates_per_node)
                    if new_nodes:
                        any_new_nodes = True
                except ValueError as e:
                    logger.warning(f"Failed to add tree layer for batch {batch_idx}: {e}")
            
            if not any_new_nodes:
                break
        
        return trees
    
    def _extract_sampling_paths_specinfer(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        tree_len: int,
        seq_lengths: torch.LongTensor,
        is_first_iteration: bool,
        temperature: float,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        # SpecInfer (arXiv 2305.09781) rejection-sampling verification for the
        # do_sample=True branch. For each batch we pick the highest-path-log-prob
        # candidate path from the tree, build sparse p_draft vectors from the
        # siblings' stored drafter probabilities, compute p_target with the
        # requested temperature, and call spec_decoding_verify.verify_path. The
        # result is (accepted prefix, resampled/bonus token), which matches the
        # shape of the greedy return tuple so the outer loop is untouched.
        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        device = logits.device
        vocab_size = logits.shape[-1]
        fallback_pos = max(0, seq_len - tree_len)
        temp = float(temperature) if temperature and temperature > 0 else 1.0

        verified_tokens_list: List[torch.Tensor] = []
        positions_list: List[torch.Tensor] = []
        llm_tokens_list: List[torch.Tensor] = []
        valid_lengths_list: List[int] = []

        for batch_idx in range(batch_size):
            actual_len = seq_lengths[batch_idx].item()
            real_fallback_pos = actual_len if is_first_iteration else fallback_pos
            tree_root_position = actual_len - 1

            node_paths = batch_node_paths[batch_idx] if batch_idx < len(batch_node_paths) else []

            # Rank candidate paths by cumulative draft log-prob, pick the top one.
            best_path: List[TreeNode] = []
            best_log = float("-inf")
            for node_path in node_paths:
                if not node_path:
                    continue
                log_p = 0.0
                ok = True
                for node in node_path:
                    if node.probability <= 0:
                        ok = False
                        break
                    log_p += math.log(node.probability)
                if not ok:
                    continue
                # Skip paths whose positions exceed the logits window.
                last_pos = node_path[-1].parent.position_in_sequence + 1
                if last_pos >= seq_len:
                    continue
                if log_p > best_log:
                    best_log = log_p
                    best_path = node_path

            if not best_path:
                # No usable path — resample one token from the fallback logits.
                final_logits = logits[batch_idx, max(0, real_fallback_pos - 1):real_fallback_pos]
                if final_logits.numel() == 0 or final_logits.shape[0] == 0:
                    final_logits = logits[batch_idx, max(0, seq_len - 1):seq_len]
                processed_logits = final_logits.clone()
                for processor in logits_processor:
                    processed_logits = processor(input_ids[batch_idx:batch_idx + 1], processed_logits)
                probs = torch.softmax(processed_logits[0] / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                verified_tokens_list.append(torch.empty(0, dtype=torch.long, device=device))
                positions_list.append(torch.tensor([tree_root_position], device=device))
                llm_tokens_list.append(torch.tensor([next_token], device=device))
                valid_lengths_list.append(0)
                continue

            path_len = len(best_path)
            target_probs = torch.zeros(path_len, vocab_size, device=device, dtype=torch.float32)
            draft_probs = torch.zeros(path_len, vocab_size, device=device, dtype=torch.float32)
            draft_tokens_t = torch.zeros(path_len, dtype=torch.long, device=device)
            path_positions: List[int] = []

            for i, node in enumerate(best_path):
                if node.parent is None or node.parent.parent is None:
                    pos = actual_len - 1 if is_first_iteration else 0
                else:
                    pos = (
                        actual_len + node.parent.position_in_sequence
                        if is_first_iteration
                        else node.parent.position_in_sequence + 1
                    )
                path_positions.append(tree_root_position + node.position_in_sequence + 1)
                row_logits = logits[batch_idx, pos].to(torch.float32)
                processed = row_logits.unsqueeze(0).clone()
                for processor in logits_processor:
                    processed = processor(input_ids[batch_idx:batch_idx + 1], processed)
                target_probs[i] = torch.softmax(processed[0] / temp, dim=-1)

                # Sparse draft distribution: siblings share the same parent, so
                # p_draft[sib.token] = sib.probability; other tokens get 0 (the
                # residual handles the rest of the vocabulary).
                siblings = node.parent.children if node.parent is not None else [node]
                for sib in siblings:
                    tok = int(sib.token_id)
                    if 0 <= tok < vocab_size:
                        draft_probs[i, tok] = max(float(sib.probability), draft_probs[i, tok].item())
                draft_tokens_t[i] = int(node.token_id)

            committed, accepted_len = verify_path(target_probs, draft_probs, draft_tokens_t)

            if accepted_len > 0:
                best_verified = committed[:accepted_len]
                best_positions = path_positions[:accepted_len]
            else:
                best_verified = []
                best_positions = []
            llm_token_val = int(committed[accepted_len]) if accepted_len < len(committed) else int(committed[-1])

            all_positions = [tree_root_position] + best_positions
            verified_tokens_list.append(
                torch.tensor(best_verified, dtype=torch.long, device=device) if best_verified
                else torch.empty(0, dtype=torch.long, device=device)
            )
            positions_list.append(torch.tensor(all_positions, dtype=torch.long, device=device))
            llm_tokens_list.append(torch.tensor([llm_token_val], dtype=torch.long, device=device))
            valid_lengths_list.append(len(best_verified))

        llm_generated_tokens = torch.stack(llm_tokens_list, dim=0)
        valid_lengths = torch.tensor(valid_lengths_list, dtype=torch.long, device=device)
        max_pos_len = max(pos.shape[0] for pos in positions_list)
        kv_cache_position_ids = torch.full((batch_size, max_pos_len), -1, dtype=torch.long, device=device)
        for i, pos in enumerate(positions_list):
            kv_cache_position_ids[i, :pos.shape[0]] = pos
        max_verified_len = max((v.shape[0] for v in verified_tokens_list), default=0)
        if max_verified_len > 0:
            verified_tokens = torch.full((batch_size, max_verified_len), -1, dtype=torch.long, device=device)
            for i, v in enumerate(verified_tokens_list):
                if v.shape[0] > 0:
                    verified_tokens[i, :v.shape[0]] = v
        else:
            verified_tokens = None
        final_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        return verified_tokens, kv_cache_position_ids, llm_generated_tokens, valid_lengths, final_positions
    def _extract_best_verified_paths_fixed(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        tree_len: int,
        seq_lengths: torch.LongTensor,
        is_first_iteration: bool,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        Returns:
            verified_tokens: [batch_size, max_verified_len] 或 None
            kv_cache_position_ids: [batch_size, max_pos_len]
            llm_generated_tokens: [batch_size, 1]
            valid_lengths: [batch_size] 每个序列验证通过的 token 数（不包括 llm token）
        """
        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        total_tree_tokens = tree_len
        fallback_pos = max(0, seq_len - total_tree_tokens)
        device = logits.device

        # Vectorize greedy verification: compute argmax per position once for the
        # whole batch, move to host, then walk tree paths in Python without per-
        # node GPU syncs. For a depth-d width-w tree this removes O(B * w^d * d)
        # .item() calls per step — the dominant CPU-GPU sync cost. Semantics are
        # preserved for do_sample=False (argmax verify equals rejection sampling
        # with a one-hot target). When do_sample=True we route through
        # spec_decoding_verify.verify_path (SpecInfer arXiv 2305.09781) below.
        if do_sample:
            return self._extract_sampling_paths_specinfer(
                logits=logits,
                batch_node_paths=batch_node_paths,
                input_ids=input_ids,
                logits_processor=logits_processor,
                tree_len=tree_len,
                seq_lengths=seq_lengths,
                is_first_iteration=is_first_iteration,
                temperature=temperature,
            )

        if logits.numel() > 0 and logits.shape[1] > 0:
            predicted_tokens_cpu = logits.argmax(dim=-1).detach().cpu().tolist()
        else:
            predicted_tokens_cpu = [[] for _ in range(batch_size)]

        # 存储结果
        verified_tokens_list = []
        positions_list = []
        llm_tokens_list = []
        valid_lengths_list = []
        final_positions_list: List[int] = []
        
        for batch_idx in range(batch_size):
            actual_len = seq_lengths[batch_idx].item()
            real_fallback_pos = actual_len if is_first_iteration else fallback_pos
            tree_root_position = actual_len - 1

            def _ensure_non_empty_logits(token_logits: torch.Tensor, *, reason: str) -> torch.Tensor:
                if token_logits.numel() > 0 and token_logits.shape[0] > 0:
                    return token_logits
                fallback_start = max(0, seq_len - 1)
                logger.warning(
                    "Speculative verification received an empty logits slice; "
                    "falling back to the last available token logits "
                    "(batch=%s reason=%s seq_len=%s fallback_start=%s)",
                    batch_idx,
                    reason,
                    seq_len,
                    fallback_start,
                )
                return logits[batch_idx, fallback_start:fallback_start + 1]
            
            node_paths = batch_node_paths[batch_idx]
            best_verified = []
            best_positions = []
            best_score = -1
            
            predicted_row = predicted_tokens_cpu[batch_idx] if batch_idx < len(predicted_tokens_cpu) else []
            for node_path in node_paths:
                verified_tokens = []
                verified_positions = []

                for node in node_path:
                    if node.parent is None or node.parent.parent is None:
                        pos = actual_len - 1 if is_first_iteration else 0
                    else:
                        pos = (
                            actual_len + node.parent.position_in_sequence
                            if is_first_iteration
                            else node.parent.position_in_sequence + 1
                        )
                    if pos >= seq_len or pos >= len(predicted_row):
                        break

                    predicted_token = predicted_row[pos]

                    if predicted_token == node.token_id:
                        verified_tokens.append(node.token_id)
                        absolute_position = tree_root_position + node.position_in_sequence + 1
                        verified_positions.append(absolute_position)
                    else:
                        break

                if len(verified_tokens) > best_score:
                    best_score = len(verified_tokens)
                    best_verified = verified_tokens
                    best_positions = verified_positions
            
            # 确定取 llm_token 的位置
            committed_pos: int  # index into logits/hidden time dim — committed endpoint
            if len(best_verified) > 0:
                pos = best_positions[-1] - tree_root_position
                if is_first_iteration:
                    pos = int(best_positions[-1])
                committed_pos = int(pos)
                final_logits = logits[batch_idx, pos].unsqueeze(0)
                final_logits = _ensure_non_empty_logits(final_logits, reason="best_verified")

                # 检查是否全 0（被裁剪），需要回退
                if torch.all(final_logits == 0):
                    # 回退：最后一个 verified token 作为 llm_token
                    llm_token = torch.tensor([best_verified[-1]], device=device)
                    best_verified = best_verified[:-1]
                    best_positions = best_positions[:-1]
                else:
                    # 正常：从 logits 采样
                    processed_logits = final_logits.clone()
                    for processor in logits_processor:
                        processed_logits = processor(
                            input_ids[batch_idx:batch_idx+1],
                            processed_logits
                        )
                    next_token = torch.argmax(processed_logits[0]).item()
                    llm_token = torch.tensor([next_token], device=device)
            else:
                # fallback: 从 fallback_pos 采样
                committed_pos = int(real_fallback_pos - 1)
                final_logits = logits[batch_idx, real_fallback_pos - 1:real_fallback_pos]
                final_logits = _ensure_non_empty_logits(final_logits, reason="fallback")
                processed_logits = final_logits.clone()
                for processor in logits_processor:
                    processed_logits = processor(
                        input_ids[batch_idx:batch_idx+1],
                        processed_logits
                    )
                next_token = torch.argmax(processed_logits[0]).item()
                llm_token = torch.tensor([next_token], device=device)
            final_positions_list.append(committed_pos)
            
            # 构建 positions
            all_positions = [tree_root_position] + best_positions
            positions = torch.tensor(all_positions, device=device)
            
            # 构建 verified_tensor
            if len(best_verified) > 0:
                verified_tensor = torch.tensor(best_verified, dtype=torch.long, device=device)
            else:
                verified_tensor = torch.empty(0, dtype=torch.long, device=device)
            
            verified_tokens_list.append(verified_tensor)
            positions_list.append(positions)
            llm_tokens_list.append(llm_token)
            valid_lengths_list.append(len(best_verified))
        
        # 统一 padding 成 batch tensor
        
        # 1. llm_generated_tokens: [batch_size, 1]
        llm_generated_tokens = torch.stack(llm_tokens_list, dim=0)
        
        # 2. valid_lengths: [batch_size]
        valid_lengths = torch.tensor(valid_lengths_list, dtype=torch.long, device=device)
        
        # 3. positions: [batch_size, max_pos_len]
        max_pos_len = max(pos.shape[0] for pos in positions_list)
        kv_cache_position_ids = torch.full(
            (batch_size, max_pos_len),
            -1,
            dtype=torch.long,
            device=device
        )
        for i, pos in enumerate(positions_list):
            kv_cache_position_ids[i, :pos.shape[0]] = pos
        
        # 4. verified_tokens: [batch_size, max_verified_len] 或 None
        max_verified_len = max(v.shape[0] for v in verified_tokens_list) if verified_tokens_list else 0
        
        if max_verified_len > 0:
            verified_tokens = torch.full(
                (batch_size, max_verified_len),
                -1,
                dtype=torch.long,
                device=device
            )
            for i, v in enumerate(verified_tokens_list):
                if v.shape[0] > 0:
                    verified_tokens[i, :v.shape[0]] = v
        else:
            verified_tokens = None

        final_positions = torch.tensor(final_positions_list, dtype=torch.long, device=device)
        return verified_tokens, kv_cache_position_ids, llm_generated_tokens, valid_lengths, final_positions
