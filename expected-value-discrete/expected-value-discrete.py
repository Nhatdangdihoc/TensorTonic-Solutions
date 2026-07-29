import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)

    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1.")

    return np.sum(x * p)