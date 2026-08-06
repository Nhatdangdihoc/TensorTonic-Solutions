import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Return torch.Tensor of the same shape as x, normalized along the last axis
    by the root mean square of x and scaled elementwise by gamma.
    """
    mean_sq = (x ** 2).mean(dim=-1, keepdim=True)
    x_hat = x / torch.sqrt(mean_sq + eps)
    return x_hat * gamma