import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    s_t = beta * s + (1 - beta) * g * g
    w_t = w - (lr / (s_t + eps)**0.5) * g
    return (w_t, s_t)
    pass