import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(1000):
        z = X @ w + b 
        p = _sigmoid(z)
        dw = (1 / n_samples) * (X.T @ (p - y))
        db = (1 / n_samples) * np.sum(p - y)
        w = w -  lr* dw
        b = b -  lr*db 
    return w,b 
    pass

  
        