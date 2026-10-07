import enum
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
    w = omega * dt
    theta = cs.sqrt(cs.sumsqr(w) + 1e-9)
    axis = w / theta
    K = skew(axis)
    return cs.MX.eye(3) + cs.sin(theta) * K + (1 - cs.cos(theta)) * (K @ K)


class WholeBodyMPC:
    def __init__(
        self,
        urdf_path,
        joints_name_list,
        n,
        dt,
        joint_damping,
        base_w,
        swing_w,
        mu=0.6,
        foot_names=('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'),
        tau_max=100.0,
        fz_max=400.0,
    ):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.tau_max = tau_max
        self.fz_max = fz_max
        self.fc_w = 1e-3
        self.q_joint_w = np.tile([30.0, 1.0, 1.0], 4)
        self.q_min = np.tile([-0.5, -0.5, -2.6], 4)
        self.q_max = np.tile([ 0.5,  1.5, -0.2], 4)
        self.nj = len(joints_name_list)
        self.nv = 6 + self.nj
        self.joint_damping = joint_damping
        self.base_z_w, self.base_R_w = base_w
        self.swing_w = swing_w
        self.foot_names = list(foot_names)

        self.kindyn = KinDynComputations(urdf_path, joints_name_list)
        self.kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

        self._warm = None          # last solution, used to warm start the next solve
        self.converged = False     # False = last result was a feasible but non-converged iterate

        self._build_dynamics_functions()
        self._build_solver()

    def _build_dynamics_functions(self):
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

        self.pf_fun = {}
        for foot in self.foot_names:
            H_sx = self.kindyn.forward_kinematics(foot, w_H_b_sx, q_sx)
            self.pf_fun[foot] = cs.Function(f'p_{foot}', [w_H_b_sx, q_sx], [H_sx[0:3, 3]])

    def _build_solver(self):
        opti = cs.Opti()
        n, nj, nv = self.n, self.nj, self.nv

        p0_p = opti.parameter(3)
        R0_p = opti.parameter(3, 3)
        qj0_p = opti.parameter(nj)
        v0_p = opti.parameter(nv)
        q_nom_p = opti.parameter(nj)
        v_base_des_p = opti.parameter(6)
        swing_vz_p = opti.parameter(4, n)
        stance_p = opti.parameter(4, n)
        vz_active_p = opti.parameter(4, n)
        fc_des_p = opti.parameter(12, n)
        swing_xy_p = opti.parameter(8, n)
        xy_active_p = opti.parameter(4, n)
        z_des_p = opti.parameter()
        R_des_p = opti.parameter(3, 3)

        self._qj_w_p = opti.parameter()
        self._vbase_w_p = opti.parameter()
        self._vj_w_p = opti.parameter()
        self._tau_w_p = opti.parameter()
        self._a_w_p = opti.parameter()

        # Stage-interleaved decision variables (required by Fatrop's structure detection)
        p, R, qj, v = [], [], [], []
        a, tau_j, Fc = [], [], []
        for k in range(n + 1):
            p.append(opti.variable(3))
            R.append(opti.variable(3, 3))
            qj.append(opti.variable(nj))
            v.append(opti.variable(nv))
            if k < n:
                a.append(opti.variable(nv))
                tau_j.append(opti.variable(nj))
                Fc.append(opti.variable(12))

        eps = 1e-3
        no_slip_M = 50.0
        cost = 0

        self._con_groups = []
        def sub(name, k, expr):
            start = opti.ng
            opti.subject_to(expr)
            self._con_groups.append((name, k, start, opti.ng))

        legs = ("FL", "FR", "BL", "BR")

        # ---- Stage 0: initial condition ----
        sub("init_p", 0, p[0] == p0_p)
        sub("init_R", 0, cs.vec(R[0] - R0_p) == 0)
        sub("init_qj", 0, qj[0] == qj0_p)
        sub("init_v", 0, v[0] == v0_p)

        for k in range(n):
            sub("dyn_p", k, p[k + 1] == p[k] + v[k][0:3] * self.dt)
            sub("dyn_R", k, R[k + 1] == so3_exp(v[k][3:6], self.dt) @ R[k])
            sub("dyn_qj", k, qj[k + 1] == qj[k] + v[k][6:] * self.dt)
            sub("dyn_v", k, v[k + 1] == v[k] + a[k] * self.dt)

            w_H_b = cs.vertcat(
                cs.horzcat(R[k], p[k]),
                cs.horzcat(cs.MX.zeros(1, 3), 1)
            )
            v_base_k = v[k][0:6]
            v_joints_k = v[k][6:]

            M = self.M_fun(w_H_b, qj[k])
            h = self.h_fun(w_H_b, qj[k], v_base_k, v_joints_k)

            contact_wrench = cs.MX.zeros(nv)
            for i, foot in enumerate(self.foot_names):
                Ji = self.J_fun[foot](w_H_b, qj[k])  # 6 x nv
                Fi = Fc[k][3 * i: 3 * i + 3]
                contact_wrench += stance_p[i, k] * (Ji[0:3, :].T @ Fi)

                if k > 0:
                    st = stance_p[i, k]
                    foot_vel = Ji[0:3, :] @ v[k]
                    # eq. 6: stance foot velocity = 0 (big-M relaxed in swing)
                    sub(f"noslip_{legs[i]}", k, opti.bounded(-no_slip_M * (1 - st), foot_vel, no_slip_M * (1 - st)))
                    # eq. 6: swing foot vertical velocity tracks the reference (relaxed in stance)
                    swing_slack = no_slip_M * (1 - vz_active_p[i, k] * (1 - st))
                    sub(f"swingvz_{legs[i]}", k, opti.bounded(-swing_slack, foot_vel[2] - swing_vz_p[i, k], swing_slack))

            # eq. 5: whole-body inverse dynamics (joint damping matches the simulator)
            net = M @ a[k] + h - contact_wrench
            sub("rnea_base", k, net[0:6] == 0)
            sub("rnea_tau", k, net[6:] + self.joint_damping * v_joints_k - tau_j[k] == 0)

            for i in range(4):
                Fi = Fc[k][3 * i: 3 * i + 3]
                st = stance_p[i, k]
                relax = self.fz_max * (1 - st)
                sub(f"fz_{legs[i]}", k, opti.bounded(-relax, Fi[2], self.fz_max))
                sub(f"fric_{legs[i]}", k, Fi[0] <= self.mu * Fi[2] + eps)
                sub(f"fric_{legs[i]}", k, Fi[0] >= -self.mu * Fi[2] - eps)
                sub(f"fric_{legs[i]}", k, Fi[1] <= self.mu * Fi[2] + eps)
                sub(f"fric_{legs[i]}", k, Fi[1] >= -self.mu * Fi[2] - eps)

            sub("tau_bound", k, opti.bounded(-self.tau_max, tau_j[k], self.tau_max))
            if k > 1:   # eq. 8 joint limits; nodes 0 and 1 are fixed by the measured q0, v0
                sub("qj_bound", k, opti.bounded(self.q_min, qj[k], self.q_max))

            # ---- Stage cost (eq. 3) ----
            cost += self._qj_w_p * cs.sumsqr(cs.sqrt(self.q_joint_w) * (qj[k] - q_nom_p))
            cost += self._vbase_w_p * cs.sumsqr(v_base_k - v_base_des_p)
            cost += self._vj_w_p * cs.sumsqr(v_joints_k)
            cost += self._tau_w_p * cs.sumsqr(tau_j[k])
            cost += self._a_w_p * cs.sumsqr(a[k])
            cost += self.fc_w * cs.sumsqr(Fc[k] - fc_des_p[:, k])
            cost += self.base_z_w * (p[k][2] - z_des_p) ** 2
            cost += self.base_R_w * cs.sumsqr(R[k] - R_des_p)
            
            if k > 0:
                for i, foot in enumerate(self.foot_names):
                    swing = (1 - stance_p[i, k]) * xy_active_p[i, k]
                    pf_xy = self.pf_fun[foot](w_H_b, qj[k])[0:2]
                    cost += self.swing_w * swing * cs.sumsqr(pf_xy - swing_xy_p[2 * i: 2*i+2, k])

        # ---- Terminal cost ----
        cost += self._qj_w_p * cs.sumsqr(cs.sqrt(self.q_joint_w) * (qj[n] - q_nom_p))
        cost += self.base_z_w * (p[n][2] - z_des_p) ** 2
        cost += self.base_R_w * cs.sumsqr(R[n] - R_des_p)

        opti.minimize(cost)

        self._solver_opts = {
            "expand": True,                   # MX graph -> SX: much cheaper Hessian/Jacobian evaluations
            "structure_detection": "auto",    # let Fatrop see the x_k / u_k stage structure
            "fatrop.print_level": 0,
            "print_time": False,
            "record_time": True,
            "fatrop.max_iter": 100,
            "fatrop.tol": 1e-3,
            "fatrop.mu_init": 0.1,
        }
        opti.solver("fatrop", self._solver_opts)
        self._struct_sig = None

        self.opti = opti
        self._p = cs.horzcat(*p)          # 3 x (n+1)
        self._qj = cs.horzcat(*qj)        # nj x (n+1)
        self._v = cs.horzcat(*v)          # nv x (n+1)
        self._tau_j = cs.horzcat(*tau_j)  # nj x n
        self._Fc_list = Fc
        self._z_des_p, self._R_des_p = z_des_p, R_des_p
        self._p0_p, self._R0_p, self._qj0_p, self._v0_p = p0_p, R0_p, qj0_p, v0_p
        self._q_nom_p, self._v_base_des_p = q_nom_p, v_base_des_p
        self._swing_vz_p, self._stance_p, self._vz_active_p = swing_vz_p, stance_p, vz_active_p
        self._fc_des_p = fc_des_p
        self._swing_xy_p = swing_xy_p
        self._xy_active_p = xy_active_p

    def constraint_report(self, top=8):
        """Rank constraint groups at the last iterate by violation (called on failure only)."""
        dbg = self.opti.debug
        g = np.array(dbg.value(self.opti.g)).flatten()
        lam = np.array(dbg.value(self.opti.lam_g)).flatten()
        lbg = np.array(dbg.value(self.opti.lbg)).flatten()
        ubg = np.array(dbg.value(self.opti.ubg)).flatten()
        viol = np.maximum(np.maximum(lbg - g, g - ubg), 0.0)
        rows = [(name, k, float(np.max(np.abs(lam[s:e]))), float(np.max(viol[s:e])))
                for name, k, s, e in self._con_groups if e > s]
        print("  [con report] largest violation (name, node, max|lam|, max viol):", flush=True)
        for r in sorted(rows, key=lambda r: -r[3])[:top]:
            print(f"    {r[0]:<14} k={r[1]:<2} lam={r[2]:.2e} viol={r[3]:.2e}", flush=True)

    def _max_violation(self, src):
        g = np.array(src.value(self.opti.g)).flatten()
        lbg = np.array(src.value(self.opti.lbg)).flatten()
        ubg = np.array(src.value(self.opti.ubg)).flatten()
        return float(np.max(np.maximum(np.maximum(lbg - g, g - ubg), 0.0)))

    def solve(
        self,
        p0,
        R0,
        qj0,
        v0,
        q_nom,
        v_base_des,
        swing_vz_schedule,
        stance_schedule,
        mpc_weights,
        vz_active_schedule,
        fc_guess,
        z_des,
        R_des,
        swing_xy,
        xy_active
    ):
        """Returns (tau_traj nj x n, qj_traj nj x (n+1), vj_traj nj x (n+1)). Raises on an infeasible result."""
        o = self.opti

        # Fatrop fixes which rows are equalities when the solver is built; the stance
        # pattern turns no-slip rows into equalities, so rebuild when the pattern changes.
        st = np.asarray(stance_schedule, dtype=float)
        va = np.asarray(vz_active_schedule, dtype=float)
        sig = tuple((int(st[:, k].sum()), int(((1 - st[:, k]) * va[:, k]).sum())) for k in range(self.n))
        if self._struct_sig is not None and sig != self._struct_sig:
            o.solver("fatrop", self._solver_opts)
        self._struct_sig = sig

        o.set_value(self._p0_p, p0)
        o.set_value(self._R0_p, R0)
        o.set_value(self._qj0_p, qj0)
        o.set_value(self._v0_p, v0)
        o.set_value(self._q_nom_p, q_nom)
        o.set_value(self._v_base_des_p, v_base_des)
        o.set_value(self._swing_vz_p, swing_vz_schedule)
        o.set_value(self._stance_p, st)
        o.set_value(self._vz_active_p, va)
        o.set_value(self._z_des_p, z_des)
        o.set_value(self._R_des_p, R_des)
        o.set_value(self._swing_xy_p, swing_xy)
        o.set_value(self._xy_active_p, xy_active)
        o.set_value(self._qj_w_p, mpc_weights[0])
        o.set_value(self._vbase_w_p, mpc_weights[1])
        o.set_value(self._vj_w_p, mpc_weights[2])
        o.set_value(self._tau_w_p, mpc_weights[3])
        o.set_value(self._a_w_p, mpc_weights[4])

        fc_des = np.asarray(fc_guess, dtype=float).reshape(12, self.n)
        o.set_value(self._fc_des_p, fc_des)

        if self._warm is not None:
            o.set_initial(self._warm)
        else:
            for k in range(self.n):
                o.set_initial(self._Fc_list[k], fc_des[:, k])

        try:
            src = o.solve()
            self.converged = True
        except RuntimeError:
            # Not certified converged: use the last iterate only if it satisfies every constraint
            src = o.debug
            if self._max_violation(src) > 1e-4:
                self.constraint_report()
                raise
            self.converged = False

        self._warm = src.value_variables()
        tau_traj = np.array(src.value(self._tau_j)).reshape(self.nj, self.n)
        qj_traj = np.array(src.value(self._qj)).reshape(self.nj, self.n + 1)
        vj_traj = np.array(src.value(self._v[6:, :])).reshape(self.nj, self.n + 1)
        return tau_traj, qj_traj, vj_traj
