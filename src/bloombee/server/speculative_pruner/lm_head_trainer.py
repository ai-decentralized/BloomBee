import os
import torch
import torch.nn.functional as F

from bloombee.server.speculative_pruner.mid_layer_LM_head import MidLMHead
from bloombee.utils.debug import dprint

CHECKPOINT_PATH = "checkpoints/lmhead/lm_head_checkpoint.pt"


class LM_head_trainer:
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        device: str = 'cuda',
        config=None,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.config = config

        # ── 用于推理 target 的 frozen 原始 LM head ─────────────
        self.original_lm_head = MidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
        self.original_lm_head.load_weight("/tmp/data/llama_weights/llama-7b-np")
        self.original_lm_head.requires_grad_(False)
        self.original_lm_head.to(dtype=torch.bfloat16)

        # ── 待训练的 LM head ────────────────────────────────────
        self.lm_head = MidLMHead(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
        self.lm_head.load_weight("/tmp/data/llama_weights/llama-7b-np")
        self.lm_head.to(dtype=torch.bfloat16)

        self.optimizer_head = torch.optim.AdamW(self.lm_head.parameters(), lr=1e-4)
        self.ite = 0

        # ── 若本地有 checkpoint 则恢复，否则从预训练权重出发 ────
        self._load_checkpoint_if_exists(CHECKPOINT_PATH)

    # ────────────────────────────────────────────────────────────
    def _load_checkpoint_if_exists(self, path: str) -> None:
        if not os.path.isfile(path):
            dprint(f"[INFO] No checkpoint found at '{path}', starting from pretrained weights.")
            return

        dprint(f"[INFO] Checkpoint found at '{path}', resuming training...")
        checkpoint = torch.load(path, map_location=self.device)
        self.lm_head.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_head.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ite = checkpoint['ite']
        dprint(f"[INFO] Resumed from iteration {self.ite}.")

    # ────────────────────────────────────────────────────────────
    def train_step(
        self,
        middle_hidden_states: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ) -> float:
        self.ite += 1

        middle_hidden_states = middle_hidden_states.to(torch.bfloat16)
        final_hidden_states  = final_hidden_states.to(torch.bfloat16)

        self.lm_head.train()
        self.lm_head.requires_grad_(True)
        self.optimizer_head.zero_grad()

        with torch.no_grad():
            target_logits = self.original_lm_head(final_hidden_states)

        with torch.enable_grad():
            mid_logits = self.lm_head(middle_hidden_states)
            b, t, v = mid_logits.shape
            mid_logits_2d    = mid_logits.view(-1, v)
            target_logits_2d = target_logits.view(-1, v)

            log_p    = F.log_softmax(mid_logits_2d.float(),    dim=-1)
            p_target = F.softmax(target_logits_2d.float(), dim=-1)
            loss     = F.kl_div(log_p, p_target, reduction='batchmean')

        loss.backward()
        self.optimizer_head.step()

        if self.ite % 1000 == 0:
            self._save_trained_head(CHECKPOINT_PATH)

        return loss.item()

    # ────────────────────────────────────────────────────────────
    def _save_trained_head(self, path: str = CHECKPOINT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'ite':                self.ite,
            'model_state_dict':   self.lm_head.state_dict(),
            'optimizer_state_dict': self.optimizer_head.state_dict(),
        }
        torch.save(checkpoint, path)
        dprint(f"[SUCCESS] Checkpoint saved to '{path}' (iteration {self.ite})")