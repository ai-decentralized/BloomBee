import torch
import torch.nn.functional as F

from bloombee.server.speculative_pruner.mid_layer_LM_head import MidLMHead

class LM_head_trainer:
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        device: str = 'cuda',
        config = None,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.config = config
        
        # LM head for getting probabilities
        self.lm_head = MidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
        # need to modify this path to real LM head path
        self.lm_head.load_weight("/tmp/data/llama_weights/llama-7b-np")
        self.lm_head.requires_grad_(False)
        self.lm_head.to(dtype=torch.bfloat16)

        
        self.original_lm_head = MidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
        # need to modify this path to real LM head path
        self.original_lm_head.load_weight("/tmp/data/llama_weights/llama-7b-np")
        self.original_lm_head.requires_grad_(False)
        self.original_lm_head.to(dtype=torch.bfloat16)
        
        self.optimizer_head = torch.optim.AdamW(
            self.lm_head.parameters(), 
            lr=1e-4
        )
        
    def train_step(
        self,
        middle_hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ) -> dict:
        self.ite = self.ite + 1
        
        middle_hidden_states = middle_hidden_states.to(torch.bfloat16)
        final_hidden_states = final_hidden_states.to(torch.bfloat16)

        self.lm_head.train()
        self.lm_head.requires_grad_(True)
        self.optimizer_head.zero_grad()

        with torch.no_grad():
            target_logits = self.original_lm_head(final_hidden_states)
            
        with torch.enable_grad():
            mid_logits = self.lm_head(middle_hidden_states) 
            b, t, v = mid_logits.shape
            mid_logits_2d = mid_logits.view(-1, v)
            target_logits_2d = target_logits.view(-1, v)

            log_p = F.log_softmax(mid_logits_2d.float(), dim=-1)
            p_target = F.softmax(target_logits_2d.float(), dim=-1)

            loss = F.kl_div(log_p, p_target, reduction='batchmean')
            

        loss.backward()
        
        self.optimizer_head.step()
        
        if self.ite % 1000 == 0:
            self._save_trained_head()
        return loss.item()
    
    def _save_trained_head(self, path="checkpoints/lmhead/lm_head_checkpoint.pt"):
        checkpoint = {
            'ite': self.ite,
            'model_state_dict': self.lm_head.state_dict(),
            'optimizer_state_dict': self.optimizer_head.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"[SUCCESS] Checkpoint saved to {path}")