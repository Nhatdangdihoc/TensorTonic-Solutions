import numpy as np

def conv2d(x, W, b):
    """
    2D convolution forward pass, channel-first, valid padding, stride=1.

    x: input,   shape (N, C_in, H, W)
    W: filters, shape (C_out, C_in, KH, KW)
    b: bias,    shape (C_out,)

    Returns y: shape (N, C_out, H_out, W_out), H_out = H-KH+1, W_out = W-KW+1
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    N, C_in, H, Win = x.shape
    C_out, C_in_w, KH, KW = W.shape
    assert C_in == C_in_w, "C_in of x and W must match"

    H_out = H - KH + 1
    W_out = Win - KW + 1

    # Build all (KH,KW) sliding windows over the H,W axes at once.
    # Resulting shape: (N, C_in, H_out, W_out, KH, KW)
    windows = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2, 3))

    # Contract over C_in, KH, KW with the filters (C_out, C_in, KH, KW)
    # windows: n i h w k l   ,   W: o i k l   ->   y: n o h w
    y = np.einsum('nihwkl,oikl->nohw', windows, W, optimize=True)

    # Add bias per output channel
    y += b.reshape(1, C_out, 1, 1)

    return y.astype(np.float64)