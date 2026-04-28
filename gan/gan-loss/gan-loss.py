import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    real_probs = np.array(real_probs, dtype=float)
    fake_probs = np.array(fake_probs, dtype=float)
    
    # Clip để tránh log(0) = -inf
    eps = 1e-8
    real_probs = np.clip(real_probs, eps, 1 - eps)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    
    loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    return round(float(loss), 4)

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    fake_probs = np.array(fake_probs, dtype=float)
    
    eps = 1e-8
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    
    loss = -np.mean(np.log(fake_probs))
    return round(float(loss), 4)