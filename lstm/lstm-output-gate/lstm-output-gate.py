import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev, x_t, C_t, W_o, b_o):
    combined = np.concatenate([h_prev, x_t], axis=-1)
    o_t = sigmoid(combined @ W_o.T + b_o)
    h_t = o_t * np.tanh(C_t)
    return (o_t, h_t)