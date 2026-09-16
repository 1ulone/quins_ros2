import os 
import numpy as np
import casadi as ca
from LOGIC.DynamicsLogic import QuadrupedDynamics

class CentroidalMPC:
    def __init__(self, urdf_path, n=10, dt=0.02, mu=0.6, mass=12.2):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.mass = mass
        self.g = np.array([0, 0, -9.81])

        self.dyn = QuadrupedDynamics(urdf_path)
        self.i_body = np.diag([1.920, 3.204, 3.921]) #from urdf (?)
        self.i_inv = np.linalg.inv(self.i_body)

        self._build_solver()

    def skew(self, v):
        return ca.vertcat(
            ca.horzcat(  0  ,-v[2], v[1]),
            ca.horzcat( v[2],  0  , -v[0]),
            ca.horzcat(-v[1], v[0],  0  )
        )

    def _build_solver(self):
        self.opti = ca.Opti()

        # Decision variables
        self.xp = self.opti.variable(3, self.n + 1)
        self.xrpy = self.opti.variable(3, self.n + 1)
        self.xv = self.opti.variable(3, self.n + 1)
        self.xw = self.opti.variable(3, self.n + 1)
        self.uf = self.opti.variable(12, self.n)

        # State / Reference parameters
        self.p_init_p = self.opti.parameter(3)
        self.v_init_p = self.opti.parameter(3)
        self.rpy_init_p = self.opti.parameter(3)
        self.w_init_p = self.opti.parameter(3)

        self.pref_p = self.opti.parameter(3)
        self.vref_p = self.opti.parameter(3)
        self.rpy_ref_p = self.opti.parameter(3)
        self.wref_p = self.opti.parameter(3)

        self.foot_pos_p = [self.opti.parameter(3) for _ in range(4)]
        self.stance_p = self.opti.parameter(4, self.n)

        # Initial condition constraints
        self.opti.subject_to(self.xp[:, 0] == self.p_init_p)
        self.opti.subject_to(self.xrpy[:, 0] == self.rpy_init_p)
        self.opti.subject_to(self.xv[:, 0] == self.v_init_p)
        self.opti.subject_to(self.xw[:, 0] == self.w_init_p)

        cost = 0
        qp = 200.0
        qv = 20.0
        qrpy = 150.0
        qw = 10.0
        rf = 1e-4

        for k in range(self.n):
          pk = self.xp[:, k]
          rpyk = self.xrpy[:, k]
          vk = self.xv[:, k]
          wk = self.xw[:, k]

          f_total = ca.vertcat(0, 0, 0)
          tau_total = ca.vertcat(0, 0, 0)

          for i in range(4):
            f_foot = self.uf[3 * i : 3 * i + 3, k]
            r_foot = self.foot_pos_p[i] - pk

            f_total += f_foot
            tau_total += self.skew(r_foot) @ f_foot

            # Friction cone & stance gating (Relaxed to prevent IPOPT singularity)
            is_st = self.stance_p[i, k]
            self.opti.subject_to(f_foot[2] >= 0.0)
            self.opti.subject_to(f_foot[2] <= 250.0 * is_st)
            self.opti.subject_to(f_foot[0] <= self.mu * f_foot[2] + 1e-3)
            self.opti.subject_to(f_foot[0] >= -self.mu * f_foot[2] - 1e-3)
            self.opti.subject_to(f_foot[1] <= self.mu * f_foot[2] + 1e-3)
            self.opti.subject_to(f_foot[1] >= -self.mu * f_foot[2] - 1e-3)

          acc = (f_total / self.mass) + self.g
          ang_acc = self.i_inv @ tau_total

          self.opti.subject_to(self.xp[:, k + 1] == pk + vk * self.dt)
          self.opti.subject_to(self.xrpy[:, k + 1] == rpyk + wk * self.dt)
          self.opti.subject_to(self.xv[:, k + 1] == vk + acc * self.dt)
          self.opti.subject_to(self.xw[:, k + 1] == wk + ang_acc * self.dt)

          # Tracking penalties
          cost += qp * ca.sumsqr(pk[0] - self.pref_p[0])
          cost += qp * ca.sumsqr(pk[1] - self.pref_p[1])
          cost += qp * ca.sumsqr(pk[2] - self.pref_p[2])
          cost += qv * ca.sumsqr(vk - self.vref_p)
          cost += qrpy * ca.sumsqr(rpyk - self.rpy_ref_p)
          cost += qw * ca.sumsqr(wk - self.wref_p)
          cost += rf * ca.sumsqr(self.uf[:, k])

        self.opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-3,
        }
        self.opti.solver("ipopt", opts)

    def solve(
          self,
          p_init,
          v_init,
          rpy_init,
          w_init,
          foot_positions,
          stance_schedule,
          p_ref,
          v_ref,
          rpy_ref=np.zeros(3),
          w_ref=np.zeros(3),
      ):
        self.opti.set_value(self.p_init_p, p_init)
        self.opti.set_value(self.v_init_p, v_init)
        self.opti.set_value(self.rpy_init_p, rpy_init)
        self.opti.set_value(self.w_init_p, w_init)

        self.opti.set_value(self.pref_p, p_ref)
        self.opti.set_value(self.vref_p, v_ref)
        self.opti.set_value(self.rpy_ref_p, rpy_ref)
        self.opti.set_value(self.wref_p, w_ref)

        for i in range(4):
            self.opti.set_value(self.foot_pos_p[i], foot_positions[i])

        self.opti.set_value(self.stance_p, np.array(stance_schedule, dtype=float).T)

        try:
            sol = self.opti.solve()
            uf_val = np.array(sol.value(self.uf)).reshape(12, self.n)
            self.last_uf = uf_val
            return uf_val[:, 0].flatten()
        except Exception:
            # If the solver fails, use the last known forces or distribute gravity evenly
            if hasattr(self, 'last_uf'):
                return self.last_uf[:, 0].flatten()
            else:
                fallback = np.zeros(12)
                fallback[2::3] = (self.mass * 9.81) / 4.0
                return fallback

if __name__ == "__main__":
    urdf_file = os.path.join(os.path.dirname(__file__), '../urdf/quadruped.urdf')
    mpc = CentroidalMPC(urdf_file)
    
    # Dummy Initial States
    p_init = np.array([0, 0, 0.5])
    v_init = np.zeros(3)
    rpy_init = np.zeros(3)
    w_init = np.zeros(3)
    
    # Dummy Foot Positions (relative to world)
    foot_positions = [
        np.array([ 0.2,  0.2, 0.0]), # LF
        np.array([ 0.2, -0.2, 0.0]), # RF
        np.array([-0.2,  0.2, 0.0]), # LH
        np.array([-0.2, -0.2, 0.0])  # RH
    ]
    
    # 4 feet on the ground for the horizon
    stance_schedule = [[True, True, True, True] for _ in range(10)]
    
    print("Setting up pure CasADi Centroidal MPC...")
    opti, U_f, p_ref, v_ref = mpc.setup_problem(p_init, v_init, rpy_init, w_init, foot_positions, stance_schedule)
    
    # Target to hover at 0.5m height
    opti.set_value(p_ref, np.array([0, 0, 0.5]))
    opti.set_value(v_ref, np.zeros(3))
    
    print("Solving OCP with IPOPT...")
    sol = opti.solve()
    
    print("MPC Solved Successfully!")
    print(f"Optimal Z-forces for 4 feet (Fc_z): {sol.value(U_f[2::3, 0])}")

