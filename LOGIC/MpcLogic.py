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

        self.i_body = np.diag([3.921, 3.204, 1.920]) #from urdf (?)
        self.i_inv = np.linalg.inv(self.i_body)

    def skew(self, v):
        return ca.vertcat(
            ca.horzcat(  0  ,-v[2], v[1]),
            ca.horzcat( v[2],  0  , v[1]),
            ca.horzcat(-v[1], v[0],  0  )
        )

    def setup_problem(self, p_init, v_init, rpy_init, w_init, foot_positions, stance_schedule):
        opti = ca.Opti()

        xp = opti.variable(3, self.n + 1)
        xrpy = opti.variable(3, self.n + 1)
        xv = opti.variable(3, self.n + 1)
        xw = opti.variable(3, self.n + 1)

        uf = opti.variable(12, self.n)

        pref = opti.parameter(3)
        vref = opti.parameter(3)

        qp = 100.0
        qv = 10.0
        rf = 1e-3

        cost = 0

        opti.subject_to(xp[:, 0]==p_init)
        opti.subject_to(xrpy[:, 0]==rpy_init)
        opti.subject_to(xv[:, 0]==v_init)
        opti.subject_to(xw[:, 0]==w_init)

        for k in range(self.n):
            pk = xp[:, k]
            rpyk = xrpy[:, k]
            vk = xv[:, k]
            wk = xw[:, k]

            f_total = ca.vertcat(0, 0, 0)
            tau_total = ca.vertcat(0, 0, 0)

            is_stance = stance_schedule[k]

            for i in range(4):
                f_foot = uf[3*i : 3*i+3, k]
                r_foot = foot_positions[i] - pk

                f_total += f_foot
                tau_total += self.skew(r_foot) @ f_foot

                if is_stance[i]:
                    opti.subject_to(f_foot[2] >= 0)
                    opti.subject_to(opti.bounded(-self.mu* f_foot[2], f_foot[0], self.mu* f_foot[2]))
                    opti.subject_to(opti.bounded(-self.mu* f_foot[2], f_foot[1], self.mu* f_foot[2]))
                else:
                    opti.subject_to(f_foot[0]==0)
                    opti.subject_to(f_foot[1]==0)
                    opti.subject_to(f_foot[2]==0)

            acc = (f_total / self.mass) + self.g
            ang_acc = self.i_inv @ tau_total

            opti.subject_to(xp[:, k+1]==pk+vk* self.dt)
            opti.subject_to(xrpy[:, k+1]==rpyk+wk* self.dt)
            opti.subject_to(xv[:, k+1]==vk+acc* self.dt)
            opti.subject_to(xw[:, k+1]==wk+ang_acc* self.dt)

            cost += qp * ca.sumsqr(pk - pref)
            cost += qv * ca.sumsqr(vk - vref)
            cost += rf * ca.sumsqr(uf[:, k])

        opti.minimize(cost)

        opts = {'ipopt.print_level':0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver('ipopt', opts)

        return opti, uf, pref, vref

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

