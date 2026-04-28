import numpy as np

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    x = np.array(x)
    W = np.array(W)
    
    logits = np.dot(x, W)  # (batch, input_dim) @ (input_dim, 1) → (batch, 1)
    prob = 1 / (1 + np.exp(-logits))
    return np.round(prob, 4)
    pass