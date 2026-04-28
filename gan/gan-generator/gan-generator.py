import numpy as np

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch, output_dim) with tanh-activated values rounded to 4 decimals
    """
    z = np.array(z)
    W = np.array(W)
    b = np.array(b)
    
    linear = np.dot(z, W) + b
    output = np.tanh(linear)
    return np.round(output, 4)
    pass