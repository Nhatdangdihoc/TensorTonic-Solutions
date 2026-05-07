import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev, x_t, W_i, b_i, W_c, b_c):
    combined = np.concatenate([h_prev, x_t], axis=-1)
    i_t = sigmoid(combined @ W_i.T + b_i)
    c_tilde = np.tanh(combined @ W_c.T + b_c)
    return (i_t, c_tilde)