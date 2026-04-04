import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (base ** (2 * (i // 2) / d_model))
            if i % 2 == 0:
                PE[pos, i] = np.sin(angle)  
            else:
                PE[pos, i] = np.cos(angle)  

    return PE
    pass