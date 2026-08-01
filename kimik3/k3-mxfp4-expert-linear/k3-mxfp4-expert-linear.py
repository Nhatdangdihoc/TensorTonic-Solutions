import torch

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _decode_expert(packed_e, scale_e, lut):
    """
    Decode one expert's packed MXFP4 weights.
    packed_e: (O, G, 16) uint8
    scale_e:  (O, G) uint8
    Returns:  (O, I) float32, where I = 32*G
    """
    O, G, _ = packed_e.shape

    # Tách nibble thấp / cao từ mỗi byte
    low = (packed_e & 0x0F).long()          # (O, G, 16)
    high = ((packed_e >> 4) & 0x0F).long()   # (O, G, 16)

    # Decode nibble -> giá trị E2M1 qua lookup table
    low_vals = lut[low]     # (O, G, 16)
    high_vals = lut[high]   # (O, G, 16)

    # Xen kẽ: nibble thấp trước, nibble cao sau -> 32 giá trị mỗi group
    interleaved = torch.stack([low_vals, high_vals], dim=-1)  # (O, G, 16, 2)
    interleaved = interleaved.reshape(O, G, 32)                # (O, G, 32)

    # Áp dụng scale E8M0 cho từng group: 2^(s-127)
    scale = torch.pow(2.0, scale_e.to(torch.float32) - 127.0)  # (O, G)
    interleaved = interleaved * scale.unsqueeze(-1)             # (O, G, 32)

    # Ghép các group lại theo chiều input -> (O, G*32) = (O, I)
    W = interleaved.reshape(O, G * 32)
    return W


def mxfp4_expert_linear(latent_tokens, packed_weights, scale_bytes,
                         selected_experts, mixture_weights, shared_output):
    """
    Returns: combined routed and shared expert output, shape (T, O),
    same dtype/device as latent_tokens.
    """
    device = latent_tokens.device
    orig_dtype = latent_tokens.dtype

    T, I = latent_tokens.shape
    K = selected_experts.shape[1]

    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)

    # Chỉ decode những expert thực sự được chọn (không đụng vào expert khác)
    unique_experts = torch.unique(selected_experts).tolist()
    decoded_cache = {}
    for e in unique_experts:
        decoded_cache[e] = _decode_expert(
            packed_weights[e].to(device),
            scale_bytes[e].to(device),
            lut,
        )  # (O, I) float32

    x = latent_tokens.to(torch.float32)          # (T, I), không sửa input gốc
    routed = torch.zeros_like(shared_output, dtype=torch.float32, device=device)

    for t in range(T):
        xt = x[t]  # (I,)
        acc = torch.zeros(routed.shape[1], dtype=torch.float32, device=device)
        for k in range(K):
            e = selected_experts[t, k].item()
            p = mixture_weights[t, k].to(torch.float32)
            We = decoded_cache[e]           # (O, I)
            acc = acc + p * (We @ xt)       # (O,)
        routed[t] = acc

    result = shared_output.to(torch.float32) + routed
    return result.to(dtype=orig_dtype)