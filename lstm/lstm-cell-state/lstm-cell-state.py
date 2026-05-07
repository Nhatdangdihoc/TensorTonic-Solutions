import numpy as np

def update_cell_state(C_prev, f_t, i_t, c_tilde):
    return f_t * C_prev + i_t * c_tilde