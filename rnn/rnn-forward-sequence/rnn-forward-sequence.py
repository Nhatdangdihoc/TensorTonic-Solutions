import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch,seq,input_dim = X.shape
    hidden_dim = h_0.shape[1]
    h_t = h_0
    hidden_states = np.zeros((batch, seq, hidden_dim))
    for t in range(seq):
        x_t = X[:,t,:]
        h_t = np.tanh(x_t @ W_xh.T + h_t @ W_hh.T + b_h)  
        hidden_states[:, t, :] = h_t 
    return hidden_states, h_t  
    pass