import numpy as np

# Step 1. Initialize dictionary and coefficient matrices
def initialise_matrices(V, k, random_state=None):
    """
    Initialize the W (basis dictionary) and H (coefficient) matrices with random non-negative values
    """
    m, n = V.shape
    rng = np.random.RandomState(random_state)
    W = rng.rand(m, k)
    H = rng.rand(k, n)
    return W, H