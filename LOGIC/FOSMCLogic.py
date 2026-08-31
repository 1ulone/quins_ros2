import numpy as np

class FractionalDerivative:
    def __init__(self, order, dt, buffer_size=100):
        self.order = order
        self.dt = dt
        self.buffer = []
        self.buffer_size = buffer_size
        self.weights = [1.0]
        for i in range(1, buffer_size):
            self.weights.append(self.weights[-1] * (1 - (order + 1) / i))
        self.weights = np.array(self.weights)

    def update(self, error):
        self.buffer.insert(0, error)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop()
        hist = np.array(self.buffer)
        n = len(hist)
        return (1.0 / (self.dt ** self.order)) * np.sum(hist * self.weights[:n].reshape(-1, 1), axis=0)

class FOSMC:
    def __init__(self, dt, lam, alpha, Ke1, Ke2, Ks, Kr, q_bound):
        self.lam = lam
        self.alpha = alpha
        self.Ke1 = Ke1
        self.Ke2 = Ke2
        self.Ks = Ks
        self.Kr = Kr
        self.frac_deriv = FractionalDerivative(order=self.alpha - 1.0, dt=dt)
        self.q_bound = q_bound

    def boundary_layer_sign(self, s):
        hs = np.sign(s)
        ss = np.sign(np.abs(s) / (np.abs(s) + self.q_bound)) * np.sign(s)
        return np.where(np.abs(s) >= self.q_bound, hs, ss)

    def compute(self, e, e_dot):
        frac_term = self.frac_deriv.update(e)
        s = e_dot + self.Ke1 * (np.sign(e) * (np.abs(e) ** self.lam)) + self.Ke2 * frac_term.flatten()
        tau_r = -self.Kr * self.boundary_layer_sign(s)
        return -self.Ks * s + tau_r
