import numpy as np

class VectorFractionalDerivative:
    def __init__(self, order, dt, buffer_size=100, dof=12):
        self.order = order
        self.dt = dt
        self.buffer = []
        self.buffer_size = buffer_size
        self.weights = [1.0]
        self.dof = dof 
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
    def __init__(self, dof, dt, lam, alpha, Ke1, Ke2, Ks, Kr, gamma_c, gamma_a, num_hidden_nodes=8):
        self.dof = dof
        self.dt = dt
        
        self.lam = lam
        self.alpha = alpha
        self.Ke1 = np.diag([Ke1] * dof)
        self.Ke2 = np.diag([Ke2] * dof)
        self.Ks = np.diag([Ks] * dof)
        self.Kr = np.diag([Kr] * dof)
        self.frac_deriv = VectorFractionalDerivative(order=self.alpha - 1.0, dt=dt, dof=dof)
        
        self.num_hidden = num_hidden_nodes
        self.gamma_c = gamma_c
        self.gamma_a = gamma_a
        self.Lambda_c = np.eye(self.num_hidden) * 0.1
        self.Lambda_a = np.eye(self.num_hidden) * 0.1
        
        # W_c correctly sized to (num_hidden, dof) to map J(t) as a vector
        # Initialized to zero to prevent massive unlearned torques on spawn
        self.W_c = np.zeros((self.num_hidden, self.dof)) 
        self.W_a = np.zeros((self.num_hidden, self.dof))
        
        self.Z_a_dim = self.dof * 4 
        self.Z_c_dim = self.dof     
        
        # Expanded RBF centers to cover full joint range (-pi to pi)
        self.c_a = np.random.uniform(-3.14, 3.14, (self.num_hidden, self.Z_a_dim))
        self.b_a = np.ones(self.num_hidden) * 2.0
        
        self.c_c = np.random.uniform(-3.14, 3.14, (self.num_hidden, self.Z_c_dim))
        self.b_c = np.ones(self.num_hidden) * 2.0
        
        self.prev_psi_c = np.zeros((self.num_hidden, 1))
        self.boundary_thickness = 0.01

    def rbf(self, z, centers, widths):
        z_expanded = np.tile(z, (self.num_hidden, 1))
        dist_sq = np.sum((z_expanded - centers) ** 2, axis=1)
        return np.exp(-dist_sq / (2 * widths ** 2)).reshape(-1, 1)

    def boundary_layer(self, s):
        # Restored original discontinuous logic to maintain baseline rigidity
        hs = np.sign(s)
        ss = np.sign(np.abs(s) / (np.abs(s) + self.boundary_thickness)) * np.sign(s)
        return np.where(np.abs(s) >= self.boundary_thickness, hs, ss)

    def compute(self, q, q_dot, q_d, q_dot_d):
        e = q - q_d 
        e_dot = q_dot - q_dot_d 

        d_alpha_e = self.frac_deriv.update(e)
        term2 = self.Ke1 @ (np.sign(e) * (np.abs(e) ** self.lam))
        term3 = self.Ke2 @ d_alpha_e
        s = e_dot + term2 + term3  
        s_norm_val = np.linalg.norm(s)

        z_a = np.concatenate((q, q_dot, e, s))
        z_c = s

        psi_a = self.rbf(z_a, self.c_a, self.b_a)
        psi_c = self.rbf(z_c, self.c_c, self.b_c)

        j_t = self.W_c.T @ psi_c 
        delta_psi_c = psi_c - self.prev_psi_c 
        self.prev_psi_c = psi_c

        o_c = np.zeros((self.dof, 1))
        
        inner_term = (self.W_c.T @ delta_psi_c) + o_c
        critic_grad = delta_psi_c @ inner_term.T + self.Lambda_c @ self.W_c
        
        self.W_c -= self.gamma_c * s_norm_val * critic_grad * self.dt

        rho = s_norm_val * j_t.flatten() + s 
        
        actor_grad = psi_a @ rho.reshape(1, -1) + s_norm_val * (self.Lambda_a @ self.W_a)
        self.W_a  -= self.gamma_a * actor_grad * self.dt

        tau_r = -self.Kr @ self.boundary_layer(s)
        tau_nn = self.W_a.T @ psi_a
        tau_p = -self.Ks @ s + tau_nn.flatten() + tau_r
        return tau_p, s
