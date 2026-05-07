import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Project: dùng np.dot theo hint
    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    # Split into heads: (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Attention scores
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    head_out = np.matmul(weights, V_heads)  # (batch, num_heads, seq_len, d_k)

    # Concat: (batch, seq_len, d_model)
    concat = head_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    return np.dot(concat, W_o)