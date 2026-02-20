from typing import Optional, Union, List, Tuple, Any

import torch
import numpy as np
import contextlib
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from bloombee.models.llama.config import DistributedLlamaConfig
from bloombee.models.llama.model import DistributedLlamaForCausalLM


from bloombee.models.llama.spe_dec_tree import SpeculativeTree, TreeNode, prepare_incremental_tree_batch

from bloombee.client.remote_generation import RemotePastKeyValues
from bloombee.client.inference_session import InferenceSession
from hivemind.utils.logging import get_logger

logger = get_logger()

class DistributedLlamaForSpeculativeGeneration(DistributedLlamaForCausalLM):
    def __init__(self, config: DistributedLlamaConfig):
        super().__init__(config)
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        streamer: Optional["BaseStreamer"] = None,
        beam_width: int = 2,
        max_tree_depth: int = 4,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 64,
        session_max_length: Optional[int] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        
        generation_config = generation_config or getattr(self, "generation_config", GenerationConfig())
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        generation_config.do_sample = False
        generation_config.return_dict_in_generate = False

        # Roll back to fixed session max length mode.
        # Keep the argument for API compatibility, but ignore runtime overrides.
        if "session_max_length" in model_kwargs:
            model_kwargs.pop("session_max_length", None)
        session_max_length = 512
        logger.info("Speculative session_max_length=%s (hardcoded)", session_max_length)

        # Use inference session for proper distributed caching
        with self.transformer.h.inference_session(max_length=session_max_length) as session:
            return self._sample_with_session(
                input_ids=input_ids,
                ssm=ssm,
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
                **model_kwargs,
            )
        
    def _sample_with_session(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        session: InferenceSession,
        streamer: Optional["BaseStreamer"],
        beam_width: int = 2,
        max_tree_depth: int = 3,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 128,
        **model_kwargs,
    ) -> torch.LongTensor:
        logger.info("Starting speculative decoding with distributed inference session!")
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
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
        initial_len = input_ids.shape[1]
        while not finished and (seq_lengths.min().item() - initial_len) < max_new_tokens:
            # 1. Build speculative trees using SSM - 传入 seq_lengths
            spec_trees = self._build_speculative_trees_batched(
                current_input_ids, ssm, beam_width, max_tree_depth, seq_lengths
            )
            
            # logger.info(f"spec_trees, {spec_trees}")
            
            # 2. Verify trees using distributed inference
            verified_tokens, verified_tokens_positions, past_key_values, llm_generated_token, valid_lengths = self._verify_trees_with_forward(
                input_ids=current_input_ids,
                llm_generated_token=llm_generated_token,
                trees=spec_trees,
                logits_processor=logits_processor,
                past_key_values=past_key_values,
                is_first_iteration=is_first_iteration,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window,
                seq_lengths=seq_lengths,
            )
            
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
            )
            
            # logger.info(f"current_input_ids: {current_input_ids}, seq_lengths: {seq_lengths}")

            if streamer is not None:
                # Stream 时根据 valid_lengths 只输出有效 token
                for i in range(batch_size):
                    if unfinished_sequences[i]:
                        if verified_tokens is not None and valid_lengths[i] > 0:
                            streamer.put(verified_tokens[i, :valid_lengths[i]].cpu())
                        streamer.put(llm_generated_token[i].cpu())

            # 5. Check if finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(current_input_ids, None)
            finished = unfinished_sequences.max() == 0
            step_idx += 1

        if streamer is not None:
            streamer.end()
        
        return current_input_ids

    def _update_input_ids_with_padding(
        self,
        current_input_ids: torch.LongTensor,
        verified_tokens: Optional[torch.LongTensor],
        llm_generated_token: torch.LongTensor,
        valid_lengths: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        pad_token_id: int,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        更新 input_ids，处理不同序列验证通过的 token 数量不同的情况
        
        Returns:
            updated_input_ids: 更新后的 input_ids，padding 对齐
            updated_seq_lengths: 更新后的每个序列真实长度
        """
        batch_size = current_input_ids.shape[0]
        device = current_input_ids.device
        
        # 计算每个序列需要添加的 token 数（verified + 1 个 llm token）
        tokens_to_add = valid_lengths + 1  # [batch_size]
        
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
            new_len = new_seq_lengths[i].item()
            
            # 复制原有的有效 token
            new_input_ids[i, :old_len] = current_input_ids[i, :old_len]
            
            # 添加验证通过的 token
            v_len = valid_lengths[i].item()
            if v_len > 0 and verified_tokens is not None:
                new_input_ids[i, old_len:old_len + v_len] = verified_tokens[i, :v_len]
                # 添加 llm_generated_token
                new_input_ids[i, old_len + v_len] = llm_generated_token[i, 0]
            else:
                # 只添加 llm_generated_token
                new_input_ids[i, old_len] = llm_generated_token[i, 0]
        
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
    ) -> Tuple[torch.LongTensor, torch.Tensor, RemotePastKeyValues, torch.Tensor, torch.Tensor]:
        """
        Verify speculative trees using standard forward() call within the active session context
        
        Returns:
            verified_tokens: [batch_size, max_verified_len] 或 None
            kv_cache_position_ids: [batch_size, max_pos_len]
            past_key_values: 更新后的 past_key_values
            llm_generated_tokens: [batch_size, 1]
            valid_lengths: [batch_size] 每个序列验证通过的 token 数
        """
        
        tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
            trees, input_ids, input_ids.device, seq_lengths=seq_lengths
        )
        
        # logger.info(f"tree_tokens: {tree_tokens}, attention_mask: {attention_mask.shape}")
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if attention_mask is None or tree_tokens.shape[1] == 0:
            logger.warning("No tree tokens to verify, falling back to regular generation")
            fallback_token = self._fallback_generation_with_forward(input_ids, logits_processor, past_key_values, seq_lengths)
            valid_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            return None, torch.zeros(batch_size, 1, dtype=torch.long, device=device), past_key_values, fallback_token, valid_lengths
        
        tree_mask_packed = self.pack_bool_mask_to_int64(attention_mask)
        
        # logger.info(f"tree_mask_packed: {tree_mask_packed}")
        
        with torch.no_grad():
            if not use_kv_cache:
                # No cache: process tree tokens directly
                logger.warning("Processing without KV cache, may cause error!!!")
                outputs = self(
                    input_ids=tree_tokens,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=False
                )
                logits = outputs.logits
                new_past_key_values = past_key_values
                
            elif is_first_iteration or past_key_values is None:
                # First iteration: process full sequence to establish cache
                # 需要根据 seq_lengths 构建正确的 full_sequence
                max_seq_len = seq_lengths.max().item()
                full_sequence = torch.cat([input_ids[:, :max_seq_len], tree_tokens], dim=-1)
                
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits
                
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
                
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits
                new_past_key_values = past_key_values
                new_past_key_values.update_seen(active_session.position)
                
        # Extract verification results - 现在返回 valid_lengths
        verified_tokens, kv_cache_position_ids, llm_generated_tokens, valid_lengths = self._extract_best_verified_paths_fixed(
            logits, batch_node_paths, input_ids, logits_processor, tree_tokens.shape[1], seq_lengths, is_first_iteration
        )
        return verified_tokens, kv_cache_position_ids, new_past_key_values, llm_generated_tokens, valid_lengths
    
    def pack_bool_mask_to_int64(self, mask_bool: torch.Tensor) -> torch.Tensor:
        assert mask_bool.dtype == torch.bool, "Input must be a bool tensor"
        return mask_bool.to(dtype=torch.int64)
    
    def _fallback_generation_with_forward(
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        seq_lengths: torch.LongTensor,
        temperature: float = 1.0
    ) -> torch.LongTensor:
        """
        Fallback to regular generation using forward() call within active session
        """
        try:
            logger.info("[DEBUG] Using fallback generation")
            
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            # 获取每个序列最后一个有效 token
            last_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            for i in range(batch_size):
                last_pos = seq_lengths[i].item() - 1
                last_tokens[i, 0] = input_ids[i, last_pos]
            
            outputs = self(
                input_ids=last_tokens,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply logits processors
            processed_logits = logits
            for processor in logits_processor:
                processed_logits = processor(input_ids, processed_logits)
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(processed_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)

            return next_token
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            eos_token_id = getattr(self.config, 'eos_token_id', 2)
            return torch.full((input_ids.shape[0], 1), eos_token_id, device=input_ids.device)
    
    def _build_speculative_trees_batched(
        self, 
        input_ids: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int, 
        max_depth: int,
        seq_lengths: torch.LongTensor,
    ) -> List[SpeculativeTree]:
        """Build speculative trees using the small model (SSM)"""
        batch_size = input_ids.shape[0]
        trees = []
        
        pad_token_id = getattr(ssm.config, 'pad_token_id', 0)

        for batch_idx in range(batch_size):
            # 获取该序列的真实长度
            actual_len = seq_lengths[batch_idx].item()
            
            # 只取有效部分的 input_ids
            valid_input_ids = input_ids[batch_idx, :actual_len]
            
            root_token = valid_input_ids[-1].item()
            tree = SpeculativeTree(root_token, f"req_{batch_idx}")
            
            for depth in range(max_depth):
                current_nodes = tree.get_nodes_at_depth(depth)
                if not current_nodes:
                    break

                # Build contexts
                contexts = []
                for node in current_nodes:
                    path_to_node = node.get_path_from_root()
                    context = torch.cat([
                        valid_input_ids[:-1],  # 使用有效的 input_ids
                        torch.tensor([root_token] + path_to_node, device=input_ids.device)
                    ])
                    contexts.append(context)

                if not contexts:
                    break

                max_len = max(len(ctx) for ctx in contexts)
                padded_contexts = []
                attention_masks = []

                for ctx in contexts:
                    pad_len = max_len - len(ctx)

                    # 左侧 padding
                    padded = torch.cat([
                        torch.full((pad_len,), pad_token_id, dtype=torch.long, device=input_ids.device),
                        ctx
                    ])

                    mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.long, device=input_ids.device),
                        torch.ones(len(ctx), dtype=torch.long, device=input_ids.device)
                    ])

                    padded_contexts.append(padded)
                    attention_masks.append(mask)

                batch_contexts = torch.stack(padded_contexts)
                batch_masks = torch.stack(attention_masks)

                # SSM forward
                with torch.no_grad():
                    # logger.info(f"batch_contexts: {batch_contexts}")
                    # logger.info(f"batch_masks: {batch_masks}")
                    outputs = ssm(batch_contexts, attention_mask=batch_masks, use_cache=False)
                    batch_logits = outputs.logits[:, -1, :]  # 左侧 padding 所以 -1 是正确的

                # Generate candidates
                candidates_per_node = []
                for i in range(len(current_nodes)):
                    logits = batch_logits[i]
                    top_k_values, top_k_indices = torch.topk(logits, k=beam_width)
                    probs = torch.softmax(logits, dim=-1)

                    candidates = []
                    for j in range(beam_width):
                        token_id = top_k_indices[j].item()
                        prob = probs[token_id].item()
                        candidates.append((token_id, prob))

                    candidates_per_node.append(candidates)

                try:
                    new_nodes = tree.add_layer(current_nodes, candidates_per_node)
                    if not new_nodes:
                        break
                except ValueError as e:
                    logger.warning(f"Failed to add tree layer: {e}")
                    break
                
            trees.append(tree)
        return trees
    
    def _extract_best_verified_paths_fixed(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        tree_len: int,
        seq_lengths: torch.LongTensor,
        is_first_iteration: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        # 存储结果
        verified_tokens_list = []
        positions_list = []
        llm_tokens_list = []
        valid_lengths_list = []
        
        for batch_idx in range(batch_size):
            actual_len = seq_lengths[batch_idx].item()
            real_fallback_pos = actual_len if is_first_iteration else fallback_pos
            tree_root_position = actual_len - 1
            
            node_paths = batch_node_paths[batch_idx]
            best_verified = []
            best_positions = []
            best_score = -1
            
            for node_path in node_paths:
                verified_tokens = []
                verified_positions = []
                
                for node in node_path:
                    pos = node.parent.position_in_sequence + 1
                    if pos >= seq_len:
                        break
                    
                    predicted_token = torch.argmax(logits[batch_idx, pos]).item()
                    
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
            if len(best_verified) > 0:
                pos = best_positions[-1] - tree_root_position
                final_logits = logits[batch_idx, pos].unsqueeze(0)
                
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
                final_logits = logits[batch_idx, real_fallback_pos - 1:real_fallback_pos]
                processed_logits = final_logits.clone()
                for processor in logits_processor:
                    processed_logits = processor(
                        input_ids[batch_idx:batch_idx+1],
                        processed_logits
                    )
                next_token = torch.argmax(processed_logits[0]).item()
                llm_token = torch.tensor([next_token], device=device)
            
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
        
        return verified_tokens, kv_cache_position_ids, llm_generated_tokens, valid_lengths
