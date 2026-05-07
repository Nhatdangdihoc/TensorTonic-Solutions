import numpy as np

def feed_forward(x, W1, b1, W2, b2):
    hidden = np.dot(x, W1) + b1        
    relu_out = np.maximum(0, hidden)    
    return np.dot(relu_out, W2) + b2   