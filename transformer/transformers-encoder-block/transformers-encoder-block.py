import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Project Q, K, V
    Q = Q @ W_q  # (batch, seq, d_model)
    K = K @ W_k
    V = V @ W_v

    # Reshape to (batch, num_heads, seq_len, d_k)
    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

    # Scaled dot-product attention
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (batch, heads, seq, seq)
    attn = softmax(scores, axis=-1)
    out = attn @ V  # (batch, heads, seq, d_k)

    # Concat heads → (batch, seq, d_model)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    return out @ W_o

def feed_forward(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
                  gamma1, beta1, gamma2, beta2, num_heads):
    # Sub-layer 1: Multi-head self-attention + residual + norm
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)

    # Sub-layer 2: FFN + residual + norm
    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_out, gamma2, beta2)

    return x