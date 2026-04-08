import torch
import torch.nn.functional as F

from bloombee.server.speculative_pruner.mid_layer_LM_head import TrainableMidLMHead, OriginalLMHead
from bloombee.utils.debug import dprint

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
        
        # 1. 中间头：使用新的 Trainable 类，不加载 30B 预训练权重
        self.lm_head = TrainableMidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
        self.lm_head.requires_grad_(True)
        self.lm_head.to(dtype=torch.bfloat16)

        # 2. 目标头：使用 Original 类，加载 30B 的真实 Head 权重
        self.original_lm_head = OriginalLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
        self.original_lm_head.load_weight("/tmp/data/llama_weights/llama-30b-np") # 确保路径正确
        self.original_lm_head.requires_grad_(False)
        self.original_lm_head.to(dtype=torch.bfloat16)
        
        self.optimizer_head = torch.optim.AdamW(
            self.lm_head.parameters(), 
            lr=1e-4
        )
        self.ite = 0
        
    def train_step(
        self,
        middle_hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
        T: float = 2.5 # 引入温度系数平滑分布
    ) -> dict:
        self.ite = self.ite + 1
        
        middle_hidden_states = middle_hidden_states.to(torch.bfloat16)
        final_hidden_states = final_hidden_states.to(torch.bfloat16)

        self.lm_head.train()
        self.optimizer_head.zero_grad()

        # 获取目标的 Soft Labels
        with torch.no_grad():
            target_logits = self.original_lm_head(final_hidden_states)
            
        with torch.enable_grad():
            mid_logits = self.lm_head(middle_hidden_states) 
            b, t, v = mid_logits.shape
            
            mid_logits_2d = mid_logits.view(-1, v)
            target_logits_2d = target_logits.view(-1, v)

            # 使用温度系数进行 KL 散度蒸馏
            log_p = F.log_softmax(mid_logits_2d.float() / T, dim=-1)
            p_target = F.softmax(target_logits_2d.float() / T, dim=-1)

            # 乘以 T 的平方，保证梯度的量级稳定
            loss = F.kl_div(log_p, p_target, reduction='batchmean') * (T * T)

        loss.backward()
        # 增加梯度裁剪，防止初期梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.lm_head.parameters(), max_norm=1.0)
        self.optimizer_head.step()
        
        if self.ite % 100 == 0:
            self._save_trained_head()
        return loss.item()
    
    def _save_trained_head(self, path="checkpoints/lmhead/lm_head_checkpoint.pt"):
        checkpoint = {
            'ite': self.ite,
            'model_state_dict': self.lm_head.state_dict(),
            'optimizer_state_dict': self.optimizer_head.state_dict(),
        }
        torch.save(checkpoint, path)
        dprint(f"[SUCCESS] Checkpoint saved to {path}")