import torch

def kv_reconstruct(c_kv: torch.Tensor, W_uk: torch.Tensor, W_uv: torch.Tensor, num_heads: int):
    """
    c_kv:  (batch, seq, d_c)
    W_uk:  (num_heads * d_nope, d_c)
    W_uv:  (num_heads * d_head, d_c)

    Returns: (K, V)
      K: (batch, heads, seq, d_nope)
      V: (batch, heads, seq, d_head)
    """
    batch, seq, d_c = c_kv.shape

    d_nope = W_uk.shape[0] // num_heads
    d_head = W_uv.shape[0] // num_heads

    # Up-project latent -> full per-head dims
    K_flat = c_kv @ W_uk.T   # (batch, seq, num_heads * d_nope)
    V_flat = c_kv @ W_uv.T   # (batch, seq, num_heads * d_head)

    # Split heads, move heads dim before seq
    K = K_flat.view(batch, seq, num_heads, d_nope).transpose(1, 2)
    V = V_flat.view(batch, seq, num_heads, d_head).transpose(1, 2)

    return K, V