import casadi as cs
import numpy as np
import adam
from adam.casadi import KinDynComputations


def skew(v):
    return cs.vertcat(
        cs.horzcat(0, -v[2], v[1]),
        cs.horzcat(v[2], 0, -v[0]),
        cs.horzcat(-v[1], v[0], 0),
    )


def so3_exp(omega, dt):
    """Rotation matrix for holding angular velocity omega constant over dt (Rodrigues' formula)."""
    w = omega * dt
    theta = cs.sqrt(cs.sumsqr(w) + 1e-9)  # eps INSIDE the sqrt -- keeps the derivative finite at w=0
    axis = w / theta
    K = skew(axis)
    return cs.MX.eye(3) + cs.sin(theta) * K + (1 - cs.cos(theta)) * (K @ K)


class WholeBodyMPC:
    def __init__(self, urdf_path, joints_name_list, n=10, dt=0.02, mu=0.6,
                 foot_names=('tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link'),
                 tau_max=1500.0, fz_max=400.0):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.tau_max = tau_max
        self.fz_max = fz_max
        self.joints_name_list = joints_name_list
        self.nj = len(joints_name_list)
        self.foot_names = list(foot_names)
        self.nv = 6 + self.nj

        self.kindyn = KinDynComputations(urdf_path, joints_name_list)
        self.kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

        self._build_dynamics_functions()
        self._build_solver()

    def _build_dynamics_functions(self):
        # adam builds its expressions in SX. Opti's variables are MX.
        # Fix: build each dynamics quantity as an SX-based Function ONCE here.
        # A Function built from SX accepts MX arguments later -- that's what
        # lets this get called inside the Opti graph without the type error.
        nj = self.nj
        w_H_b_sx = cs.SX.sym('w_H_b', 4, 4)
        q_sx = cs.SX.sym('q', nj)
        v_base_sx = cs.SX.sym('v_base', 6)
        v_joints_sx = cs.SX.sym('v_joints', nj)

        M_sx = self.kindyn.mass_matrix(w_H_b_sx, q_sx)
        h_sx = self.kindyn.bias_force(w_H_b_sx, q_sx, v_base_sx, v_joints_sx)

        self.M_fun = cs.Function('M_fun', [w_H_b_sx, q_sx], [M_sx])
        self.h_fun = cs.Function('h_fun', [w_H_b_sx, q_sx, v_base_sx, v_joints_sx], [h_sx])

        self.J_fun = {}
        for foot in self.foot_names:
            J_sx = self.kindyn.jacobian(foot, w_H_b_sx, q_sx)
            self.J_fun[foot] = cs.Function(f'J_{foot}', [w_H_b_sx, q_sx], [J_sx])

    def _build_solver(self):
        opti = cs.Opti()
        n, nj, nv = self.n, self.nj, self.nv

        p = opti.variable(3, n + 1)
        R = [opti.variable(3, 3) for _ in range(n + 1)]
        qj = opti.variable(nj, n + 1)
        v = opti.variable(nv, n + 1)
        a = opti.variable(nv, n)
        tau_j = opti.variable(nj, n)
        Fc = opti.variable(12, n)

        p0_p = opti.parameter(3)
        R0_p = opti.parameter(3, 3)
        qj0_p = opti.parameter(nj)
        v0_p = opti.parameter(nv)
        qj_ref_p = opti.parameter(nj, n + 1)
        v_ref_p = opti.parameter(nj, n + 1)
        stance_p = opti.parameter(4, n)

        opti.subject_to(p[:, 0] == p0_p)
        opti.subject_to(cs.vec(R[0] - R0_p) == 0)
        opti.subject_to(qj[:, 0] == qj0_p)
        opti.subject_to(v[:, 0] == v0_p)

        qj_w, vj_w, tau_w, a_w = 80.0, 2.0, 1e-3, 1e-4
        eps = 1e-3
        no_slip_M = 50.0  # big-M slack: only relevant while a foot is swinging
        cost = 0

        for k in range(n):
            w_H_b = cs.vertcat(
                cs.horzcat(R[k], p[:, k]),
                cs.horzcat(cs.MX.zeros(1, 3), 1)
            )
            q_joints_k = qj[:, k]
            v_base_k = v[0:6, k]
            v_joints_k = v[6:, k]

            M = self.M_fun(w_H_b, q_joints_k)
            h = self.h_fun(w_H_b, q_joints_k, v_base_k, v_joints_k)

            contact_wrench = cs.MX.zeros(nv)
            for i, foot in enumerate(self.foot_names):
                Ji = self.J_fun[foot](w_H_b, q_joints_k)  # 6 x nv
                Fi = Fc[3 * i: 3 * i + 3, k]
                contact_wrench += Ji[0:3, :].T @ Fi

                # NEW: stance feet don't move. Swing feet are unconstrained here.
                st = stance_p[i, k]
                foot_vel = Ji[0:3, :] @ v[:, k]
                opti.subject_to(opti.bounded(-no_slip_M * (1 - st), foot_vel, no_slip_M * (1 - st)))

            net = M @ a[:, k] + h - contact_wrench
            opti.subject_to(net[0:6] == 0)
            opti.subject_to(net[6:] == tau_j[:, k])

            opti.subject_to(v[:, k + 1] == v[:, k] + a[:, k] * self.dt)
            opti.subject_to(p[:, k + 1] == p[:, k] + v[0:3, k] * self.dt)
            opti.subject_to(cs.vec(R[k + 1] - R[k] @ so3_exp(v[3:6, k], self.dt)) == 0)
            opti.subject_to(qj[:, k + 1] == qj[:, k] + v[6:, k] * self.dt)

            for i in range(4):
                Fi = Fc[3 * i: 3 * i + 3, k]
                st = stance_p[i, k]
                opti.subject_to(Fi[2] >= 0.0)
                opti.subject_to(Fi[2] <= self.fz_max * st)
                opti.subject_to(Fi[0] <= self.mu * Fi[2] + eps)
                opti.subject_to(Fi[0] >= -self.mu * Fi[2] - eps)
                opti.subject_to(Fi[1] <= self.mu * Fi[2] + eps)
                opti.subject_to(Fi[1] >= -self.mu * Fi[2] - eps)

            opti.subject_to(opti.bounded(-self.tau_max, tau_j[:, k], self.tau_max))

            cost += qj_w * cs.sumsqr(q_joints_k - qj_ref_p[:, k])
            cost += vj_w * cs.sumsqr(v_joints_k - v_ref_p[:, k])
            cost += tau_w * cs.sumsqr(tau_j[:, k])
            cost += a_w * cs.sumsqr(a[:, k])

        cost += qj_w * cs.sumsqr(qj[:, n] - qj_ref_p[:, n])

        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                               "ipopt.max_iter": 200, "ipopt.tol": 1e-3})

        self.opti = opti
        self._p, self._R, self._qj, self._v, self._a, self._tau_j, self._Fc = p, R, qj, v, a, tau_j, Fc
        self._p0_p, self._R0_p, self._qj0_p, self._v0_p = p0_p, R0_p, qj0_p, v0_p
        self._qj_ref_p, self._v_ref_p, self._stance_p = qj_ref_p, v_ref_p, stance_p

    def solve(self, p0, R0, qj0, v0, qj_ref, v_ref, stance_schedule):
        self.opti.set_value(self._p0_p, p0)
        self.opti.set_value(self._R0_p, R0)
        self.opti.set_value(self._qj0_p, qj0)
        self.opti.set_value(self._v0_p, v0)
        self.opti.set_value(self._qj_ref_p, qj_ref)
        self.opti.set_value(self._v_ref_p, v_ref)
        self.opti.set_value(self._stance_p, stance_schedule)
        try:
            sol = self.opti.solve()
            self._last_sol = sol
            tau0 = np.array(sol.value(self._tau_j[:, 0])).flatten()
            qj_1 = np.array(sol.value(self._qj[:, 1])).flatten()
            vj_1 = np.array(sol.value(self._v[6:, 1])).flatten()
            self._last = (tau0, qj_1, vj_1)
            return tau0, qj_1, vj_1
        except Exception as e:
            print(f"[WholeBodyMPC] solve failed, holding last solution: {e}")
            if hasattr(self, "_last"):
                return self._last
            return np.zeros(self.nj), np.array(qj_ref)[:, 1], np.array(v_ref)[:, 1]


if __name__ == "__main__":
    import os

    LEG_NAMES = ['FL', 'FR', 'BL', 'BR']
    JOINT_NAMES = {
        'FL': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
        'FR': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
        'BL': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
        'BR': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint'],
    }
    joints_name_list = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]

    urdf_path = os.path.join(os.path.dirname(__file__), '../urdf/quadruped.urdf')
    mpc = WholeBodyMPC(urdf_path, joints_name_list, n=10, dt=0.02)

    idle_angles = [0.00, 0.45, -0.90]
    qj0 = np.tile(idle_angles, 4)

    p0 = np.array([0.0, 0.0, 0.0401])
    R0 = np.eye(3)
    v0 = np.zeros(mpc.nv)
    qj_ref = np.tile(qj0.reshape(-1, 1), (1, mpc.n + 1))
    v_ref = np.zeros((mpc.nj, mpc.n + 1))
    stance = np.ones((4, mpc.n))

    print("Building and solving one WholeBodyMPC step...")
    tau0, qj_1, vj_1 = mpc.solve(p0, R0, qj0, v0, qj_ref, v_ref, stance)
    print(f"tau0: {tau0}")
    Fc0 = np.array(mpc._last_sol.value(mpc._Fc[:, 0])).reshape(4, 3)
    a0 = np.array(mpc._last_sol.value(mpc._a[:, 0])).flatten()
    print(f"Fc per foot (node 0): {Fc0}")
    print(f"Total vertical force: {Fc0[:,2].sum():.2f} N  (expect roughly 12.2 * 9.81 = {12.2*9.81:.1f} N)")
    print(f"Base + joint acceleration (node 0): {a0}")
    print("y-acceleration across the horizon:", 
      [round(float(mpc._last_sol.value(mpc._a[1, k])), 4) for k in range(mpc.n)])
