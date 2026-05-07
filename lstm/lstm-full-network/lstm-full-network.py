import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        N, T, _ = X.shape
        h = np.zeros((N, self.hidden_dim))
        C = np.zeros((N, self.hidden_dim))
        y_all = []
    
        for t in range(T):
            combined = np.concatenate([h, X[:, t, :]], axis=-1)
            f_t = sigmoid(combined @ self.W_f.T + self.b_f)
            i_t = sigmoid(combined @ self.W_i.T + self.b_i)
            c_tilde = np.tanh(combined @ self.W_c.T + self.b_c)
            o_t = sigmoid(combined @ self.W_o.T + self.b_o)
            C = f_t * C + i_t * c_tilde
            h = o_t * np.tanh(C)
            y_all.append(h @ self.W_y.T + self.b_y)  # (N, output_dim)
    
        y = np.stack(y_all, axis=1)  # (N, T, output_dim)
        return (y, h, C)