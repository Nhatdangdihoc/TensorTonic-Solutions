import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x, dtype=float)
    
    if rng is None:
        random_vals = np.random.random(x.shape)
    else:
        random_vals = rng.random(x.shape)
    
    scale = 1 / (1 - p)
    dropout_pattern = np.where(random_vals < (1 - p), scale, 0)
    
    output = x * dropout_pattern
    
    return output, dropout_pattern
    pass