import torch
import torch.nn.functional as F


def coarse_moe(x, gate_weight, W_gate_ffn, W_up_ffn, W_down_ffn, top_k):
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    num_experts = gate_weight.shape[0]
    d_inter = W_gate_ffn.shape[0]
    expert_dim = d_inter // num_experts   # equal-size chunks

    # --- Router: sigmoid scores -> top-k -> renormalize ---
    logits = x @ gate_weight.t()                        # (batch, seq, num_experts)
    scores = torch.sigmoid(logits)
    topk_scores, topk_idx = torch.topk(scores, top_k, dim=-1)      # (batch, seq, top_k)
    topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    # --- Experts: contiguous slices of the shared FFN, each a SwiGLU ---
    out = torch.zeros_like(x)
    for e in range(num_experts):
        lo, hi = e * expert_dim, (e + 1) * expert_dim
        Wg, Wu, Wd = W_gate_ffn[lo:hi], W_up_ffn[lo:hi], W_down_ffn[:, lo:hi]

        gate = F.silu(x @ Wg.t())          # (batch, seq, expert_dim)
        up = x @ Wu.t()                    # (batch, seq, expert_dim)
        expert_out = (gate * up) @ Wd.t()  # (batch, seq, d_model)

        # weight is 0 for tokens that didn't route to this expert
        w = (topk_weights * (topk_idx == e)).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
        out = out + w * expert_out

    return out