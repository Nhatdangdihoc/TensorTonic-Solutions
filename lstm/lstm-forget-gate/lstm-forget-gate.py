import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev, x_t, W_f, b_f):
    combined = np.concatenate([h_prev, x_t], axis=-1)
    return sigmoid(combined @ W_f.T + b_f)