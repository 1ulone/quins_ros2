import casadi as cs
import numpy as np
import adam
from adam.casadi import KinDynComputations

DEFAULT_WEIGHTS = {
    "base_lin": 100.0,     # body position task
    "base_ang": 100.0,     # body rotation task
    "swing": 100.0,        # swing foot task
    "contact": 1.0e4,      # stance no-slip hardness (bigger = harder)
    "contact_kd": 20.0,    # pulls leftover stance foot velocity back to zero
    "posture": 0.01,        # keeps joints near the posture reference
    "tau": 1.0e-4,         # small torque penalty
    "fc": 1.0e-2,          # keeps contact forces near the weight-split guess
}
WEIGHT_KEYS = list(DEFAULT_WEIGHTS)

class WholeBodyController:
    def __init__(
        self,
        urdf_path,
        joints_name_list,
        joint_damping,
        mu=0.6,
        foot_names=('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'),
        tau_max=100.0,
        fz_max=400.0,
    ):
        self.mu = mu
        self.tau_max = tau_max
        self.fz_max = fz_max
        self.joint_damping = joint_damping
        self.foot_names = list(foot_names)
        self.nj = len(joints_name_list)     # 12 joints
        self.nv = 6 + self.nj               # 18 = 6 base + 12 joints
        self.nf = len(self.foot_names)      # 4 feet

        self.kindyn = KinDynComputations(urdf_path, joints_name_list)
        # MIXED: base velocity = [linear (world), angular (world)], same as the MPC used
        self.kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

        self._build_dynamics_functions()
        self._build_qp()   # Piece 2

        self.ok = False
        self.last = {
            "tau": np.zeros(self.nj),
            "a": np.zeros(self.nv),
            "Fc": np.zeros(3 * self.nf),
            "acc_base": np.zeros(6),
            "acc_foot": np.zeros(3 * self.nf),
        } 

    def _build_dynamics_functions(self):
        nj = self.nj
        w_H_b = cs.SX.sym('w_H_b', 4, 4)
        q = cs.SX.sym('q', nj)
        v_b = cs.SX.sym('v_base', 6)
        v_j = cs.SX.sym('v_joints', nj)
        v = cs.vertcat(v_b, v_j)

        M = self.kindyn.mass_matrix(w_H_b, q)
        h = self.kindyn.bias_force(w_H_b, q, v_b, v_j)

        Jc, Jdv, pf = [], [], []
        for foot in self.foot_names:
            # 3 x 18, linear part only
            J = self.kindyn.jacobian(foot, w_H_b, q)[0:3, :]

            # 3 x 18
            Jd = self.kindyn.jacobian_dot(foot, w_H_b, q, v_b, v_j)[0:3, :]
            Jc.append(J)

            # 3 x 1
            Jdv.append(Jd @ v)

            # 3 x 1, world position
            pf.append(self.kindyn.forward_kinematics(foot, w_H_b, q)[0:3, 3])

        # One call returns everything the QP needs for this instant
        self.dyn_fun = cs.Function(
            'wbc_dyn',
            [w_H_b, q, v_b, v_j],
            [M, h, cs.vertcat(*Jc), cs.vertcat(*Jdv), cs.vertcat(*pf)],
            ['w_H_b', 'q', 'v_base', 'v_joints'],
            ['M', 'h', 'Jc', 'Jdv', 'pf'],
        )

    def _build_qp(self):
        opti = cs.Opti('conic')
        nv, nj, nf = self.nv, self.nj, self.nf

        a = opti.variable(nv)
        Fc = opti.variable(3 * nf)
        tau = opti.variable(nj)

        M = opti.parameter(nv, nv)
        h = opti.parameter(nv)
        Jc = opti.parameter(3 * nf, nv)
        Jdv = opti.parameter(3 * nf)
        v = opti.parameter(nv)
        stance = opti.parameter(nf)
        acc_base = opti.parameter(6)
        acc_foot = opti.parameter(3 * nf)
        a_joint_ref = opti.parameter(nj)
        fc_ref = opti.parameter(3 * nf)
        w = opti.parameter(len(WEIGHT_KEYS))
        W = {k: w[i] for i, k in enumerate(WEIGHT_KEYS)}

        cost = W["base_lin"] * cs.sumsqr(a[0:3] - acc_base[0:3])
        cost += W["base_ang"] * cs.sumsqr(a[3:6] - acc_base[3:6])

        foot_acc = Jc @ a + Jdv
        foot_vel = Jc @ v
        for i in range(nf):
            r = slice(3 * i, 3 * i + 3)
            st = stance[i]
            cost += W["swing"] * (1 - st) * cs.sumsqr(foot_acc[r] - acc_foot[r])
            cost += W["contact"] * st * cs.sumsqr(foot_acc[r] + W["contact_kd"] * foot_vel[r])

        cost += W["posture"] * cs.sumsqr(a[6:] - a_joint_ref)
        cost += W["tau"] * cs.sumsqr(tau)
        cost += W["fc"] * cs.sumsqr(Fc - fc_ref)
        opti.minimize(cost)

        net = M @ a + h - Jc.T @ Fc
        opti.subject_to(net[0:6] == 0)
        opti.subject_to(net[6:] + self.joint_damping * v[6:] - tau == 0)
        opti.subject_to(opti.bounded(-self.tau_max, tau, self.tau_max))

        for i in range(nf):
            fx, fy, fz = Fc[3 * i], Fc[3 * i  + 1], Fc[3 * i + 2]
            opti.subject_to(fz >= 0)
            opti.subject_to(fz <= self.fz_max * stance[i])
            opti.subject_to(fx - self.mu * fz <= 0)
            opti.subject_to(fx + self.mu * fz >= 0)
            opti.subject_to(fy - self.mu * fz <= 0)
            opti.subject_to(fy + self.mu * fz >= 0)

        opti.solver('qpoases', {"printLevel": "none", "error_on_fail": True})

        self.qp_fun = opti.to_function(
            'wbc_qp',
            [M, h, Jc, Jdv, v, stance, acc_base, acc_foot, a_joint_ref, fc_ref, w],
            [a, Fc, tau],
        )

    def update_state(self, p, R, q, v_base, v_joints):
        w_H_b = np.eye(4)
        w_H_b[0:3, 0:3] = R
        w_H_b[0:3, 3] = p
        M, h, Jc, Jdv, pf = self.dyn_fun(w_H_b, q, v_base, v_joints)

        self.M = np.array(M)
        self.h = np.array(h).flatten()
        self.Jc = np.array(Jc)
        self.Jdv = np.array(Jdv).flatten()
        self.pf = np.array(pf).flatten()
        self.v = np.concatenate([v_base, v_joints])
        self.vf = self.Jc @ self.v

    def solve(self, stance, acc_base, acc_foot, a_joint_ref, fc_ref, weights=None):
        W = dict(DEFAULT_WEIGHTS)
        if weights:
            W.update(weights)
        w = np.array([W[k] for k in WEIGHT_KEYS])

        try:
            a, Fc, tau = self.qp_fun(
                self.M, self.h, self.Jc, self.Jdv, self.v,
                np.asarray(stance, dtype=float), acc_base, acc_foot,
                a_joint_ref, fc_ref, w,
            )
        except RuntimeError as e:
            self.ok = False
            print(f"[wbc] QP failed, reusing last torques: {e}", flush=True)
            return self.last

        a = np.array(a).flatten()
        self.ok = True
        self.last = {
            "tau": np.array(tau).flatten(),
            "a": a,
            "Fc": np.array(Fc).flatten(),
            "acc_base": a[0:6].copy(),
            "acc_foot": self.Jc @ a + self.Jdv,
        }

        return self.last
