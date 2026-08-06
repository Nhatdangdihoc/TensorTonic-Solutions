import torch

def rope_freqs(n_tokens: int, head_dim: int, rope_theta: float) -> tuple:
    """
    Return tuple (cos, sin), each a torch.Tensor of shape (n_tokens, head_dim // 2).
    """
    half_dim = head_dim // 2
    i = torch.arange(0, half_dim, dtype=torch.float32)
    theta = 1.0 / (rope_theta ** (2 * i / head_dim))

    positions = torch.arange(n_tokens, dtype=torch.float32)
    angles = torch.outer(positions, theta)

    return torch.cos(angles), torch.sin(angles)