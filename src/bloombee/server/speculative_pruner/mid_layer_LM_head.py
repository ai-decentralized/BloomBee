import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class SimpleLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def load_weight(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "lm_head.weight")
        w = np.load(path)
        self.weight.data.copy_(torch.from_numpy(w).to(torch.float32))
        print(f"[OK] Initialized with pretrained weights: {self.weight.shape}")

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight)