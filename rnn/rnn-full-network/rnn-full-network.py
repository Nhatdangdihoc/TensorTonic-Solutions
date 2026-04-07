import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """    
        N, T, input_dim = X.shape
        if h_0 is None:
            h_0 = np.zeros((N, self.hidden_dim))
        h_t = h_0
        hidden_states = []
    
        for t in range(T):
            x_t = X[:, t, :]
            h_t = np.tanh(x_t @ self.W_xh.T + h_t @ self.W_hh.T + self.b_h)
            hidden_states.append(h_t)
        all_h = np.stack(hidden_states, axis=1)
        y_seq = all_h.reshape(N * T, self.hidden_dim) @ self.W_hy.T + self.b_y
        y_seq = y_seq.reshape(N, T, -1)
        return y_seq, h_t
        pass