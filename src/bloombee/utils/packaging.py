import torch
def normalize_arg(x):
    if x is None:
        return torch.tensor([], dtype=torch.float32)
    return x
