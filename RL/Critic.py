import numpy as np

def g_cost(s, eps, xi):
    # NOTE:
    # Farid Paper Eq. 35: 
    # 0 inside the dead zone, g_s * s^2 up to xi, capped at xi.
    # g_s= 1 / xi makes g coninuous at |s| = xi (g_s * xi^2 = xi)

    a = np.abs(s)
    return np.where(a <= eps, 0.0, np.where(a <= xi, a ** 2 / xi, xi))

class RBFCritic:
    # NOTE: Radial Basis Function Neural Network (Farid Paper Eq. 27-28, Eq. 38):
    # J_hat(z) = Wc^T psi(z)
    # Trained on the TD error (eq. 40-41) with sigma modification (eq. 42)

    def __init__(self, centers, gamma=0.98, lr=0.2, lr_min=0.02, lr_tau=250, sigma=1e-4):
        self.c = np.asarray(centers, dtype=float)
        d = np.linalg.norm(self.c[:, None] - self.c[None], axis=2)
        np.fill_diagonal(d, np.inf)
        # width: mean distance to the 2 nearest centers (Moody and Darken, 1989), so each unit stays local.
        # The median over all pairs gives ~0.6 activation everywhere in 23-D, and the units become copies of the bias
        self.b = float(np.mean(np.sort(d, axis=1)[:, :2]))
        self.w = np.zeros(len(self.c) + 1)
        self.gamma = gamma
        self.sigma = sigma
        self.lr = lr
        self.lr_min = lr_min
        self.lr_tau = lr_tau
        self.n = 0
        
    def features(self, z):
        # NOTE: Eq. 27
        d2 = np.sum((self.c - z) ** 2, axis=1)
        return np.append(np.exp(-d2 / (2.0 * self.b ** 2)), 1.0)

    def value(self, z):
        # NOTE: Eq. 38
        return float(self.w @ self.features(z))

    def update(self, z_prev, cost, z, k_s=1.0, terminal_value=None, learn=True):
        # NOTE: Eq. 36 and 40: TD error, J(t - T) should equal cost over [t - T, t] + gamma * J(t)
        psi_prev = self.features(z_prev)
        j_next = terminal_value if terminal_value is not None else self.value(z)
        delta = float(self.w @ psi_prev) - (cost + self.gamma * j_next)
        if not learn:
            return delta

        # NOTE: eq. 42 kept (step on Uc = delta^2 / 2, scaled by ||s||, plus sigma-modification),
        # but as semi-gradient TD(0): only J(t - T) moves toward its target, J(t) is held fixed.
        # Farid's residual form also moves J(t); at T = 0.02 s both states are almost the same,
        # so the two moves cancel and the critic barely learns the cost level.
        lr = max(self.lr / (1.0 + self.n / self.lr_tau), self.lr_min)   # decaying step; the floor keeps it learning online
        self.n += 1
        self.w -= lr * (k_s * delta * psi_prev / (1.0 + psi_prev @ psi_prev) + self.sigma * self.w)
        if not np.all(np.isfinite(self.w)) or np.max(np.abs(self.w)) > 1e6:
            raise FloatingPointError(f"critic diverged: delta={delta:.3g} cost={cost:.3g} k_s={k_s:.3g} "
                                     f"|psi|^2={float(psi_prev @ psi_prev):.3g} |z_prev|={np.linalg.norm(z_prev):.3g}")
        return delta
