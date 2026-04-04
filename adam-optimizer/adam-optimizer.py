import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.array(param, dtype=float)
    grad  = np.array(grad,  dtype=float)
    m     = np.array(m,     dtype=float)
    v     = np.array(v,     dtype=float)
    m_new = beta1 * m + (1 - beta1) * grad          # Cập nhật moment bậc 1
    v_new = beta2 * v + (1 - beta2) * grad * grad   # Cập nhật moment bậc 2

    mt_hat = m_new / (1 - beta1**t)                 # Bias correction
    vt_hat = v_new / (1 - beta2**t)                 # Bias correction

    param_new = param - lr * mt_hat / (vt_hat**0.5 + eps)  # Cập nhật param

    return param_new, m_new, v_new
    pass