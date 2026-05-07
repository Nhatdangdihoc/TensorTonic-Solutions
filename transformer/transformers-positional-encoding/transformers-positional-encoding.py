import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    PE = np.zeros((seq_length, d_model))
    pos = np.arange(seq_length).reshape(-1, 1)
    i = np.arange(0, d_model, 2)
    div_term = np.power(10000, i / d_model)
    PE[:, 0::2] = np.sin(pos / div_term)
    PE[:, 1::2] = np.cos(pos / div_term)
    return PE