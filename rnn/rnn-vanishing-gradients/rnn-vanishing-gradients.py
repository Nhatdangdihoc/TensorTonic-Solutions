import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    norm = 1.0  
    res = []
    for i in range(T):
        res.append(norm)
        norm = norm * spectral_norm  
    
    return res
    pass