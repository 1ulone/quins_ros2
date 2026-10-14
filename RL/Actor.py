import numpy as np

class RBFActor:
    # NOTE: the actor will output a :
    # shift of pre-squash gains around the baseline, u = u0 + W_a^T psi(z) + noise.
    # Trained with the one-step actor-critic rule for a Gaussian Policy

    def __init__(
        self,
        critic,
        n_act,
        lr=0.01,
        explore=0.1, 
        leak=1e-3,
        shift_max=3.0,
        seed=1,
    ):
        self.critic = critic
        self.w = np.zeros((len(critic.c)+1, n_act))
        self.w0 = self.w.copy()
        self.lr = lr
        self.explore = explore
        self.leak = leak
        self.shift_max = shift_max
        self.rng = np.random.default_rng(seed)


    def shift(self, z):
        # NOTE: learned shift of pre-squash gain, capped to the no gain parks on bound
        return np.clip(self.critic.features(z) @ self.w, -self.shift_max, self.shift_max)

    def act(self, z, u0):
        eps = self.rng.standard_normal(self.w.shape[1])
        return u0 + self.shift(z) + self.explore * eps, eps

    def update(self, z_prev, delta, eps):
        psi = self.critic.features(z_prev)
        self.w += self.lr * delta * np.outer(psi, eps) / (1.0 + psi @ psi)
        self.w -= self.lr * self.leak * (self.w - self.w0)
