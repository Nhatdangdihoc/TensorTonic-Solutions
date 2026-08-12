import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    tb = np.mean(x)
    d = (1/(len(x)-1))* (np.sum((x-tb)**2))
    return d,np.sqrt(d)
    
   
   
