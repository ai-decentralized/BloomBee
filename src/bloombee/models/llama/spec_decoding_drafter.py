import torch
from bloombee.models.llama.spe_dec_tree import SpeculativeTree
from typing import List
import threading
from transformers.cache_utils import DynamicCache


class MultiSSMDrafter:
    
    def __init__(self, ssm_model_name: str, num_workers: int = 2, device: str = 'cuda'):
        from transformers import AutoModelForCausalLM
        
        self.num_workers = num_workers
        self.device = torch.device(device)
        
        self.ssms = []
        self.streams = []
        for _ in range(num_workers):
            ssm = AutoModelForCausalLM.from_pretrained(ssm_model_name)
            ssm = ssm.to(self.device)
            ssm.eval()
            self.ssms.append(ssm)
            self.streams.append(torch.cuda.Stream(device=self.device))
        
        with torch.no_grad():
            dummy = torch.ones(1, 8, dtype=torch.long, device=self.device)
            for ssm in self.ssms:
                ssm(dummy, attention_mask=torch.ones_like(dummy))
    
    def build_trees_parallel(
        self,
        input_ids: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        beam_width: int,
        max_depth: int,
    ) -> List:
        
        batch_size = input_ids.shape[0]
        chunk_size = (batch_size + self.num_workers - 1) // self.num_workers
        
        all_results = [None] * batch_size
        
        def worker_fn(worker_idx: int, batch_indices: List[int]):
            ssm = self.ssms[worker_idx]
            stream = self.streams[worker_idx]
            
            with torch.cuda.stream(stream):
                results = self._build_trees_batched(
                    batch_indices, input_ids, seq_lengths, ssm, beam_width, max_depth
                )
                for batch_idx, tree in results:
                    all_results[batch_idx] = tree
        
        # 启动线程
        threads = []
        for worker_idx in range(self.num_workers):
            start = worker_idx * chunk_size
            end = min(start + chunk_size, batch_size)
            if start < batch_size:
                t = threading.Thread(target=worker_fn, args=(worker_idx, list(range(start, end))))
                threads.append(t)
                t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 同步所有 streams
        for stream in self.streams:
            stream.synchronize()
        
        return all_results

    def _build_trees_batched(
        self,
        batch_indices: List[int],
        input_ids: torch.LongTensor,
        seq_lengths: torch.LongTensor,
        ssm,
        beam_width: int,
        max_depth: int,
    ) -> List:
        
        pad_token_id = getattr(ssm.config, 'pad_token_id', 0)
        
        trees = {}
        valid_inputs = {}
        prefix_lengths = {}
        
        for batch_idx in batch_indices:
            actual_len = seq_lengths[batch_idx].item()
            valid_input_ids = input_ids[batch_idx, :actual_len]
            valid_inputs[batch_idx] = valid_input_ids
            prefix_lengths[batch_idx] = max(actual_len - 1, 0)
            
            root_token = valid_input_ids[-1].item()
            trees[batch_idx] = SpeculativeTree(root_token, f"req_{batch_idx}")
        
        # ========== 预计算 prefix cache ==========
        max_prefix_len = max(prefix_lengths.values())
        
        if max_prefix_len == 0:
            prefix_cache = None
        else:
            padded_prefixes = []
            prefix_masks = []
            
            for batch_idx in batch_indices:
                pf_len = prefix_lengths[batch_idx]
                
                if pf_len > 0:
                    prefix = valid_inputs[batch_idx][:-1]
                else:
                    prefix = torch.tensor([], dtype=torch.long, device=self.device)
                
                pad_len = max_prefix_len - pf_len
                
                if pf_len > 0:
                    padded_prefixes.append(torch.cat([
                        torch.full((pad_len,), pad_token_id, dtype=torch.long, device=self.device),
                        prefix
                    ]))
                else:
                    padded_prefixes.append(
                        torch.full((max_prefix_len,), pad_token_id, dtype=torch.long, device=self.device)
                    )
                
                prefix_masks.append(torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=self.device),
                    torch.ones(pf_len, dtype=torch.long, device=self.device)
                ]))
            
            batch_prefixes = torch.stack(padded_prefixes)
            batch_prefix_masks = torch.stack(prefix_masks)
            
            with torch.no_grad():
                prefix_outputs = ssm(batch_prefixes, attention_mask=batch_prefix_masks, use_cache=True)
                prefix_cache = prefix_outputs.past_key_values  # 这是 DynamicCache 对象
        
        idx_map = {batch_idx: i for i, batch_idx in enumerate(batch_indices)}
        
        # ========== 按 depth 扩展 ==========
        for depth in range(max_depth):
            
            all_paths = []
            node_mapping = []
            cache_indices = []
            
            for batch_idx in batch_indices:
                tree = trees[batch_idx]
                root_token = valid_inputs[batch_idx][-1].item()
                
                for node in tree.get_nodes_at_depth(depth):
                    path = node.get_path_from_root()
                    path_tokens = torch.tensor([root_token] + path, dtype=torch.long, device=self.device)
                    all_paths.append(path_tokens)
                    node_mapping.append((batch_idx, node))
                    cache_indices.append(idx_map[batch_idx])
            
            if not all_paths:
                break
            
            max_path_len = max(len(p) for p in all_paths)
            max_pf_len = max(prefix_lengths[nm[0]] for nm in node_mapping)
            
            padded_paths = []
            path_masks = []
            
            for i, path in enumerate(all_paths):
                batch_idx = node_mapping[i][0]
                pf_len = prefix_lengths[batch_idx]
                path_len = len(path)
                pad_len = max_path_len - path_len
                prefix_pad_len = max_pf_len - pf_len
                
                padded_paths.append(torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=torch.long, device=self.device),
                    path
                ]))
                path_masks.append(torch.cat([
                    torch.zeros(prefix_pad_len, dtype=torch.long, device=self.device),
                    torch.ones(pf_len, dtype=torch.long, device=self.device),
                    torch.zeros(pad_len, dtype=torch.long, device=self.device),
                    torch.ones(path_len, dtype=torch.long, device=self.device)
                ]))
            
            batch_paths = torch.stack(padded_paths)
            batch_path_masks = torch.stack(path_masks)
            
            # 构建新的 DynamicCache，按 cache_indices 选取
            if prefix_cache is not None:
                node_cache = DynamicCache()
                for layer_idx in range(len(prefix_cache)):
                    key, value = prefix_cache[layer_idx]
                    node_cache.update(
                        key[cache_indices],
                        value[cache_indices],
                        layer_idx
                    )
            else:
                node_cache = None
            
            with torch.no_grad():
                outputs = ssm(
                    batch_paths,
                    attention_mask=batch_path_masks,
                    past_key_values=node_cache,
                    use_cache=False
                )
                all_logits = outputs.logits[:, -1, :]
            
            batch_node_results = {}
            for i, (batch_idx, node) in enumerate(node_mapping):
                logits = all_logits[i]
                _, top_k_indices = torch.topk(logits, k=beam_width)
                probs = torch.softmax(logits, dim=-1)
                
                candidates = [(top_k_indices[j].item(), probs[top_k_indices[j]].item()) 
                            for j in range(beam_width)]
                
                if batch_idx not in batch_node_results:
                    batch_node_results[batch_idx] = []
                batch_node_results[batch_idx].append((node, candidates))
            
            any_new = False
            for batch_idx in batch_indices:
                if batch_idx not in batch_node_results:
                    continue
                
                tree = trees[batch_idx]
                nodes = [nc[0] for nc in batch_node_results[batch_idx]]
                candidates = [nc[1] for nc in batch_node_results[batch_idx]]
                
                try:
                    if tree.add_layer(nodes, candidates):
                        any_new = True
                except ValueError:
                    pass
            
            if not any_new:
                break
        
        return [(idx, trees[idx]) for idx in batch_indices]