import torch
import math

def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def ffn(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
        W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    hidden = gelu(x @ W1.T + b1)   # (seq_len, d) @ (d, 4d) → (seq_len, 4d)
    out    = hidden @ W2.T + b2     # (seq_len, 4d) @ (4d, d) → (seq_len, d)
    return out