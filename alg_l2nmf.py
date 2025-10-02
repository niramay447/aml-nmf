import numpy as np
from init_random import initialise_matrices as random_init
from init_nndsvd import nndsvd_init

def H_update(V, W, H):
    """
    Multiplicative Update Rules to solve for H by setting fixed W
    H_new = H_old*(W^T*V / (W^T*W*H_old))
    """
    numerator = np.dot(np.transpose(W), V)
    denominator = np.dot(np.dot(np.transpose(W), W), H)
    H_new = H * (numerator / denominator)
    return H_new

def W_update(V, W, H):
    """
    Multiplicative Update Rules to solve for W by setting fixed H
    W_new = W_old*(V*H^T / (W_old*H*H^T))
    """
    numerator = np.dot(V, np.transpose(H))
    denominator = np.dot(np.dot(W, H), np.transpose(H))
    W_new = W * (numerator / denominator)
    return W_new

def l2_nmf(V, k, max_iter=1000, tol=1e-6, verbose=True, random_state=None, init="random"):
    """
    Complete optimisation process. 'init' can be 'random' or 'nndsvd'.
    """
    # initialise W, H
    if init == "nndsvd":
        W, H = nndsvd_init(V, k, random_state=random_state)
    else:
        W, H = random_init(V, k, random_state)

    # main loop
    for iteration in range(max_iter):
        H_new = H_update(V, W, H)
        # Update W
        W_new = W_update(V, W, H_new)
        # Calculate error
        error_H = np.sqrt(np.sum((H_new - H)**2, axis=(0, 1))) / H.size
        error_W = np.sqrt(np.sum((W_new - W)**2, axis=(0, 1))) / W.size

        # Check convergence
        if error_H < tol and error_W < tol:
            if verbose:
                print(f"Converged at iteration {iteration+1}")
            H, W = H_new, W_new
            break
        
        # Print progress every 100 iterations
        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}: Error H = {error_H:.6f},  Error W = {error_W:.6f}")

        H, W = H_new, W_new

    return W, H
