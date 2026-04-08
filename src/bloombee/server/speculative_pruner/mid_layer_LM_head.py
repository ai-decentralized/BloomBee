import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from bloombee.utils.debug import dprint

# 目标层：保持原样，只做前向推理，提供 Soft Labels
class OriginalLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def load_weight(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "lm_head.weight")
        w = np.load(path)
        self.weight.data.copy_(torch.from_numpy(w).to(torch.float32))
        dprint(f"[OK] Original LM Head initialized: {self.weight.shape}")

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight)

# 中间层：加厚网络，引入非线性，强制随机初始化
class TrainableMidLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, mid_dim=2048):
        super().__init__()
        # 使用 MLP 结构增强特征提取能力
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid_dim, bias=False),
            nn.SiLU(), 
            nn.LayerNorm(mid_dim), 
            nn.Linear(mid_dim, vocab_size, bias=False)
        )
        self._init_weights()

    def _init_weights(self):
        # 强制随机初始化，抹除错误的先验知识
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        dprint(f"[OK] Trainable Mid LM Head randomly initialized.")

    def forward(self, hidden_states):
        return self.net(hidden_states)