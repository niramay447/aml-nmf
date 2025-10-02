# init_nndsvd.py
import numpy as np

class NNDSVD:
    def __init__(self, tiny=1e-11):
        self.tiny = tiny

    def initialize(self, V, k, random_state=None):
        U, S, VT = np.linalg.svd(V, full_matrices=False)
        U, S, VT = U[:, :k], S[:k], VT[:k, :]
        E = VT.T

        m, n = V.shape
        W = np.zeros((m, k), dtype=V.dtype)
        H = np.zeros((k, n), dtype=V.dtype)

        # first component (nonnegative)
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(E[:, 0])

        # remaining components
        for j in range(1, k):
            u = U[:, j]
            v = E[:, j]

            up = np.where(u >= 0, u, 0.0)
            un = np.where(u < 0, -u, 0.0)
            vp = np.where(v >= 0, v, 0.0)
            vn = np.where(v < 0, -v, 0.0)

            nup, nun = np.linalg.norm(up, 2), np.linalg.norm(un, 2)
            nvp, nvn = np.linalg.norm(vp, 2), np.linalg.norm(vn, 2)

            termp = nup * nvp
            termn = nun * nvn

            if termp >= termn:
                if nup > 0:
                    W[:, j] = (np.sqrt(S[j] * termp) / nup) * up
                if nvp > 0:
                    H[j, :] = (np.sqrt(S[j] * termp) / nvp) * vp
            else:
                if nun > 0:
                    W[:, j] = (np.sqrt(S[j] * termn) / nun) * un
                if nvn > 0:
                    H[j, :] = (np.sqrt(S[j] * termn) / nvn) * vn

        W[W < self.tiny] = 0.0
        H[H < self.tiny] = 0.0
        return W, H

    def __call__(self, V, k, random_state=None):
        return self.initialize(V, k, random_state)

def nndsvd_init(V, k, tiny=1e-11, random_state=None):
    return NNDSVD(tiny=tiny).initialize(V, k, random_state)