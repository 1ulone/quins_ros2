import numpy as np

# Task rows per gain group (layout set by TaskLogic: body xyz | body rotation | feet)
GROUPS = {
    "base_lin": slice(0, 3),
    "base_ang": slice(3, 6),
    "swing":    slice(6, 18),
}

# Starting gains. With lam = 1, Ke2 = 0, Kr = 0 this reduces to a PD with
# Kp = Ke1 * Ks and Kd = Ke1 + Ks (close to the PD gains that already walk)
# Ke2 found by test (lam = 1, Kr off): rotation fell at 2.0, swing got worse at 1.5 and fell at 3.0
DEFAULT_GAINS = {
    "base_lin": {"Ke1": 8.0,  "Ke2": 1.0,  "Ks": 8.0,  "Kr": 0.5},
    "base_ang": {"Ke1": 14.0, "Ke2": 1.0,  "Ks": 14.0, "Kr": 1.0},
    "swing":    {"Ke1": 30.0, "Ke2": 0.75, "Ks": 30.0, "Kr": 2.0},
}
GAIN_NAMES = ("Ke1", "Ke2", "Ks", "Kr")


def gl_weights(order, n):
    # NOTE: Grunwald-Letnikov weights: w0 = 1, wj = w(j-1) * (1 - (order + 1) / j)
    w = np.ones(n)
    for j in range(1, n):
        w[j] = w[j - 1] * (1.0 - (order + 1.0) / j)
    return w


class FOSMC:
    """Task-space fractional-order sliding mode controller.
    In: task error e, its rate e_dot, feed-forward acceleration. Out: commanded task acceleration."""

    def __init__(self, dof, dt, lam=0.95, alpha=1.5, memory=100, eps=0.01, delta=0.5, gains=None):
        self.dof = dof
        self.dt = dt
        self.memory = memory
        self.eps = eps          # keeps (|e| + eps)^(lam - 1) finite at e = 0
        self.delta = delta      # boundary layer width (eq. 51)

        self.hist = np.zeros((memory, dof))     # error history, row 0 = newest
        self.hist_v = np.zeros((memory, dof))
        self.last_in = np.zeros(dof)
        self.prev_active = np.zeros(dof, dtype=bool)
        self.s = np.zeros(dof)

        self.groups = {g: dict(v) for g, v in DEFAULT_GAINS.items()}
        if gains:
            for g, vals in gains.items():
                self.groups[g].update(vals)
        self._expand_gains()

        self.lam = lam
        self.set_alpha(alpha)

    def _expand_gains(self):
        # Group gains -> one value per task row (self.Ke1, self.Ke2, self.Ks, self.Kr)
        for name in GAIN_NAMES:
            vec = np.zeros(self.dof)
            for g, rows in GROUPS.items():
                vec[rows] = self.groups[g][name]
            setattr(self, name, vec)

    def set_alpha(self, alpha):
        self.alpha = alpha
        # D^(alpha-1) weights, used on e (surface) and on e_dot (output: D^alpha e = D^(alpha-1) e_dot)
        self.w_s = gl_weights(alpha - 1.0, self.memory) / self.dt ** (alpha - 1.0)

    def update_gains(self, groups=None, lam=None, alpha=None):
        """groups: {"swing": {"Ks": 40.0}, ...}, partial updates are fine."""
        if groups:
            for g, vals in groups.items():
                self.groups[g].update(vals)
            self._expand_gains()
        if lam is not None:
            self.lam = lam
        if alpha is not None and alpha != self.alpha:
            self.set_alpha(alpha)   # history is kept, only the weights are rebuilt

    def compute(self, e, e_dot, acc_ff, active=None, freeze=None):
        if active is None:
            active = np.ones(self.dof, dtype=bool)
        if freeze is None or self.alpha >= 1.0:
            freeze = np.zeros(self.dof, dtype=bool)

        # Anti-windup: frozen rows repeat their previous input, so the memory stops growing
        e_in = np.where(freeze, self.last_in, e)
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = e_in
        self.hist_v = np.roll(self.hist_v, 1, axis=0)
        self.hist_v[0] = e_dot

        # Rows that just became active (lift-off, first call): fresh memory at the current
        # error and error rate, as if both had always been that value
        started = active & ~self.prev_active
        if np.any(started):
            self.hist[:, started] = e[started]
            self.hist_v[:, started] = e_dot[started]
            e_in[started] = e[started]
        self.prev_active = active.copy()
        self.last_in = e_in

        d_s = self.w_s @ self.hist      # D^(alpha-1) e
        # D^alpha e as D^(alpha-1) of the measured e_dot (Caputo form): same value, but it no
        # longer differentiates the error history, whose targets carry velocity jumps
        d_o = self.w_s @ self.hist_v

        # eq. 31: sliding surface
        sig = np.sign(e) * np.abs(e) ** self.lam
        s = e_dot + self.Ke1 * sig + self.Ke2 * d_s
        self.s = s

        D = np.where(np.abs(s) >= self.delta, np.sign(s), s / (np.abs(s) + self.delta))

        return (acc_ff
                - self.lam * self.Ke1 * (np.abs(e) + self.eps) ** (self.lam - 1.0) * e_dot
                - self.Ke2 * d_o
                - self.Ks * s
                - self.Kr * D)
