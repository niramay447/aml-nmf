# from alg_l1rrnmf import l1_rrnmf

# U, V, E = l1_rrnmf(X=V_noise, k=40, lam=0.05, max_iter=5000, tol=1e-5,
#                    init="nndsvd", random_state=42, verbose=True, log_every=100)

# X_clean_est = U @ V
# X_noisy_est = U @ V + E
import numpy as np
from init_random import initialise_matrices as random_init
from init_nndsvd import nndsvd_init

# --- helpers -----------------------------------------------------------------

def init_uv(X, k, init="nndsvd", random_state=None):
    if init == "nndsvd":
        U, V = nndsvd_init(X, k, random_state=random_state)
    else:
        U, V = random_init(X, k, random_state)
    return U.astype(np.float64, copy=False), V.astype(np.float64, copy=False)

def rel_change(A_new, A_old, eps=1e-8):
    return np.linalg.norm(A_new - A_old) / (np.linalg.norm(A_old) + eps)

def _objective_L1(X, U, V, E, lam):
    R = X - (U @ V + E)
    return 0.5 * np.sum(R*R) + lam * np.sum(np.abs(E))

# --- updates -----------------------------------------------------------------

def update_U(X, U, V, E, eps=1e-8):
    # U *= ((X - E) V^T) / (U (V V^T))
    X_hat = X - E
    num = X_hat @ V.T
    den = (U @ (V @ V.T)) + eps
    U *= num / den
    np.maximum(U, 0.0, out=U)
    return U

def update_VE_stacked(X, U, V, E, lam, eps=1e-8, eta=0.25, max_bt=5):
    """
    Sign-safe, projected update for [V; E+; E-] with simple backtracking.

    U_hat = [[ U,  I,          -I        ],
             [ 0, sqrt(lam) I,  sqrt(lam) I ]]
    V_hat = [ V; E+; E- ],   X_tilde = [ X; 0_{m x n} ]
    """
    m, n = X.shape
    k = V.shape[0]
    I = np.eye(m)

    tiny = 1e-12
    Epos = np.maximum(E, 0.0)
    Eneg = np.maximum(-E, 0.0)
    Vhat = np.vstack((V, Epos + tiny, Eneg + tiny))          # (k+2m) x n

    top = np.hstack((U, I, -I))                              # m x (k+2m)
    bot = np.hstack((np.zeros((m, k)),
                     np.sqrt(lam) * I,
                     np.sqrt(lam) * I))                      # m x (k+2m)
    Uhat   = np.vstack((top, bot))                           # (2m) x (k+2m)
    Xtilde = np.vstack((X, np.zeros((m, n))))

    # precompute terms
    AV = Uhat.T @ (Uhat @ Vhat)
    B  = Uhat.T @ Xtilde
    G  = AV - B
    S  = np.abs(AV) + eps

    step = G / S
    np.clip(step, -5.0, 5.0, out=step)   # tame extreme ratios

    base_obj = _objective_L1(X, U, V, E, lam)
    cur_eta = eta

    for _ in range(max_bt):
        Vtrial = Vhat - cur_eta * Vhat * step
        np.maximum(Vtrial, tiny, out=Vtrial)

        V_try    = Vtrial[:k, :]
        Epos_try = Vtrial[k:k+m, :]
        Eneg_try = Vtrial[k+m:, :]
        E_try    = Epos_try - Eneg_try

        obj = _objective_L1(X, U, V_try, E_try, lam)
        if obj <= base_obj + 1e-10:  # accept
            Vhat = Vtrial
            V[:, :] = V_try
            E[:, :] = E_try
            return V, E
        cur_eta *= 0.5  # backtrack

    # fallback: very small step
    Vhat = Vhat - 0.05 * Vhat * step
    np.maximum(Vhat, tiny, out=Vhat)
    V[:, :] = Vhat[:k, :]
    E[:, :] = Vhat[k:k+m, :] - Vhat[k+m:, :]
    return V, E

def l1_rrnmf(X, k, lam=0.02, max_iter=4000, tol=1e-5, eps=1e-8,
             init="nndsvd", random_state=None, verbose=True, log_every=100, eta=0.25):
    """
    L1-Regularized Robust NMF with explicit sparse noise E.
    Model: X ≈ U V + E, with E = E⁺ - E⁻ (E⁺,E⁻ ≥ 0)
    Minimizes: 0.5||X - (U V + E)||_F^2 + λ ||E||_1
    Returns: U (m×k), V (k×n), E (m×n)
    """
    X = X.astype(np.float64, copy=False)
    m, n = X.shape

    # init U, V
    U, V = init_uv(X, k, init=init, random_state=random_state)

    # init E > 0 so it can grow
    rng = np.random.RandomState(random_state)
    E = 1e-6 * rng.rand(m, n)

    for it in range(max_iter):
        U_old, V_old, E_old = U.copy(), V.copy(), E.copy()

        # U update + column normalization (rescale V)
        U = update_U(X, U, V, E, eps=eps)
        col = np.linalg.norm(U, axis=0) + 1e-12
        U /= col
        V *= col[:, None]

        # stable stacked step for V and E (with backtracking)
        V, E = update_VE_stacked(X, U, V, E, lam=lam, eps=eps, eta=eta)

        # convergence + safety
        dU = rel_change(U, U_old, eps)
        dV = rel_change(V, V_old, eps)
        dE = rel_change(E, E_old, eps)
        d  = max(dU, dV, dE)

        if verbose and ((it + 1) % log_every == 0 or it == 0):
            print(f"Iteration {it+1}: ΔU={dU:.3e} ΔV={dV:.3e} ΔE={dE:.3e}")

        if np.isnan(d) or d > 1e3:
            if verbose:
                print("Early stop: divergence detected.")
            break

        if d < tol:
            if verbose:
                print(f"Converged at iteration {it+1}")
            break

    return U, V, E
