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
        q_nom_p = opti.parameter(nj)
        v_base_des_p = opti.parameter(6)
        swing_vz_p = opti.parameter(4, n)
        stance_p = opti.parameter(4, n)

        opti.subject_to(p[:, 0] == p0_p)
        opti.subject_to(cs.vec(R[0] - R0_p) == 0)
        opti.subject_to(qj[:, 0] == qj0_p)
        opti.subject_to(v[:, 0] == v0_p)

        qj_w, vj_w, tau_w, a_w = 1.0, 0.05, 1e-3, 1e-4
        vbase_w = 20.0
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

                # stance feet don't move -- but only enforce this on PREDICTED
                # (free) velocity nodes. v[:,0] is pinned to the measured v0_p,
                # which is never exactly zero in a live sim, so constraining
                # foot_vel to 0 there is infeasible by construction.
                if k > 0:
                    st = stance_p[i, k]
                    foot_vel = Ji[0:3, :] @ v[:, k]
                    opti.subject_to(opti.bounded(-no_slip_M * (1 - st), foot_vel, no_slip_M * (1 - st)))

                    # Swing constraint, eq. 6: vertical foot velocity HARD-tracks the
                    # commanded profile while swinging (st=0); inactive while in stance
                    # (st=1, big-M slack). Horizontal directions stay unconstrained --
                    # the MPC still optimizes step location itself, per the paper.
                    opti.subject_to(opti.bounded(
                        -no_slip_M * st, foot_vel[2] - swing_vz_p[i, k], no_slip_M * st
                    ))

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

            cost += qj_w * cs.sumsqr(q_joints_k - q_nom_p)
            cost += vbase_w * cs.sumsqr(v_base_k - v_base_des_p)
            cost += vj_w * cs.sumsqr(v_joints_k)
            cost += tau_w * cs.sumsqr(tau_j[:, k])
            cost += a_w * cs.sumsqr(a[:, k])

        cost += qj_w * cs.sumsqr(qj[:, n] - q_nom_p)

        opti.minimize(cost)
        # opti.solver("ipopt", {
        #     "ipopt.print_level": 0, "print_time": 0,
        #     "ipopt.max_iter": 200, "ipopt.tol": 1e-3,
        #     "ipopt.warm_start_init_point": "yes"
        # })

        opti.solver("fatrop", {
            "fatrop.print_level": 0,
            "fatrop.max_iter": 30,
            "fatrop.tol": 1e-3,
            "fatrop.mu_init": 0.1,
            # "jit": True,
            # "compiler": "shell",
            # "jit_options": {"flags": ["-O0", "-fPIC"]}
        })

        self.opti = opti
        self._p, self._R, self._qj, self._v, self._a, self._tau_j, self._Fc = p, R, qj, v, a, tau_j, Fc
        self._p0_p, self._R0_p, self._qj0_p, self._v0_p = p0_p, R0_p, qj0_p, v0_p
        self._q_nom_p, self._v_base_des_p, self._swing_vz_p, self._stance_p = (
            q_nom_p, v_base_des_p, swing_vz_p, stance_p
        )

    def solve(self, p0, R0, qj0, v0, q_nom, v_base_des, swing_vz_schedule, stance_schedule):
        # If the previous solve's background thread is still alive, DO NOT touch
        # self.opti at all -- not set_value, not solve(). That's the actual bug:
        # a timed-out thread keeps running, and starting a second solve on the
        # same object while it's still live corrupts shared solver memory.
        if hasattr(self, "_solve_thread") and self._solve_thread.is_alive():
            if hasattr(self, "_last"):
                return self._last
            return (np.zeros((self.nj, self.n)),
                    np.tile(np.array(qj0, dtype=float).reshape(-1, 1), (1, self.n + 1)),
                    np.zeros((self.nj, self.n + 1)))

        self.opti.set_value(self._p0_p, p0)
        self.opti.set_value(self._R0_p, R0)
        self.opti.set_value(self._qj0_p, qj0)
        self.opti.set_value(self._v0_p, v0)
        self.opti.set_value(self._q_nom_p, q_nom)
        self.opti.set_value(self._v_base_des_p, v_base_des)
        self.opti.set_value(self._swing_vz_p, swing_vz_schedule)
        self.opti.set_value(self._stance_p, stance_schedule)

        # Guard against feeding the solver a state that's already diverged.
        # This is what actually caused the segfault: FOSMC over-corrected for
        # a near-zero tau0 on the previous tick, threw the base height down to
        # 0.08m within one cycle, and the NEXT solve NaN'd on that extreme
        # configuration -- a hard native crash, not something a timeout or
        # try/except can catch. Refuse the solve before that ever reaches
        # Fatrop, and hold the last known-good output instead.
        qj0_arr = np.array(qj0, dtype=float)
        v0_arr = np.array(v0, dtype=float)
        p0_arr = np.array(p0, dtype=float)
        sane = (
            np.all(np.isfinite(qj0_arr)) and np.all(np.isfinite(v0_arr)) and np.all(np.isfinite(p0_arr))
            and -0.05 < p0_arr[2] < 2.00          # wide enough to cover the spawn drop (z~1.5)
            and np.max(np.abs(v0_arr)) < 50.0
        )
        if not sane:
            print(f"[WholeBodyMPC] refusing solve: state outside sane envelope "
                  f"(base z={p0_arr[2]:.3f}, max|v0|={np.max(np.abs(v0_arr)):.1f})", flush=True)
            # Don't keep feeding it a stale WALK command -- that's what turned one
            # bad tick into a full collapse last time. Give it a safe target to
            # recover toward (upright, zero torque, hold nominal pose) instead.
            safe_qj = np.tile(np.array(q_nom, dtype=float).reshape(-1, 1), (1, self.n + 1))
            return np.zeros((self.nj, self.n)), safe_qj, np.zeros((self.nj, self.n + 1))

        if hasattr(self, "_last_sol"):
            try:
                self.opti.set_initial(self._last_sol.value_variables())
                self.opti.set_initial(self.opti.lam_g, self._last_sol.value(self.opti.lam_g))
            except Exception:
                pass
        import threading
        result = {}
        def _run_solve():
            try:
                result["sol"] = self.opti.solve()
            except Exception as e:
                result["exc"] = e
        self._solve_thread = threading.Thread(target=_run_solve, daemon=True)
        self._solve_thread.start()
        self._solve_thread.join(timeout=2.0)
        if self._solve_thread.is_alive():
            # Solver is stuck (confirmed: happens on specific stance/swing
            # transitions, reproduced independently). Don't block mpc_worker
            # forever -- fall back and let the NEXT tick (with updated,
            # possibly non-pathological data) try again.
            print("[WholeBodyMPC] solve TIMED OUT (>2s), abandoning this attempt", flush=True)
            if hasattr(self, "_last"):
                return self._last
            return np.zeros((self.nj, self.n)), np.tile(np.array(qj0).reshape(-1,1), (1,self.n+1)), np.zeros((self.nj, self.n+1))
        if "exc" in result:
            raise result["exc"]
        try:
            sol = result["sol"]
            self._last_sol = sol
            # Return the FULL horizon, not one node. The caller interpolates this
            # by elapsed wall-clock time, per the paper's 80Hz-solve/500Hz-interpolate
            # scheme -- picking a single fixed node is what caused the frozen-then-jumps
            # behavior before.
            tau_traj = np.array(sol.value(self._tau_j)).reshape(self.nj, self.n)
            qj_traj = np.array(sol.value(self._qj)).reshape(self.nj, self.n + 1)
            v_traj = np.array(sol.value(self._v[6:, :])).reshape(self.nj, self.n + 1)
            self._last = (tau_traj, qj_traj, v_traj)
            return tau_traj, qj_traj, v_traj
        except Exception as e:
            print(f"[WholeBodyMPC] solve failed: {e}", flush=True)
            try:
                dbg = self.opti.debug
                k = 1  # first free horizon node -- where a bad transition would show up
                p1 = np.array(dbg.value(self._p[:, k])).flatten()
                R1 = np.array(dbg.value(self._R[k]))
                qj1 = np.array(dbg.value(self._qj[:, k])).flatten()
                v1 = np.array(dbg.value(self._v[:, k])).flatten()
                Fc0 = np.array(dbg.value(self._Fc[:, 0])).reshape(4, 3)
                tau0_dbg = np.array(dbg.value(self._tau_j[:, 0])).flatten()
                stance0 = np.array(dbg.value(self._stance_p[:, 0])).flatten()
                print(f"[DEBUG] last iterate, node k={k}:", flush=True)
                print(f"  stance flags (node 0): {stance0}", flush=True)
                print(f"  base p: {np.round(p1,4)}", flush=True)
                print(f"  qj: {np.round(qj1,4)}", flush=True)
                print(f"  v: {np.round(v1,4)}", flush=True)
                print(f"  Fc per foot (fx,fy,fz): {np.round(Fc0,3)}", flush=True)
                print(f"  tau0: {np.round(tau0_dbg,3)}  (tau_max={self.tau_max})", flush=True)
                w_H_b1 = np.eye(4); w_H_b1[0:3, 0:3] = R1; w_H_b1[0:3, 3] = p1
                for i, foot in enumerate(self.foot_names):
                    Ji = np.array(self.J_fun[foot](w_H_b1, qj1))
                    fv = Ji[0:3, :] @ v1
                    print(f"  {foot}: foot_vel={np.round(fv,4)}  Fc_z={Fc0[i,2]:.2f}  mu*Fz={self.mu*Fc0[i,2]:.2f} vs Fx,Fy={Fc0[i,0]:.2f},{Fc0[i,1]:.2f}", flush=True)
            except Exception as e2:
                print(f"[DEBUG] could not extract debug values: {e2}", flush=True)
            if hasattr(self, "_last"):
                return self._last
            return np.zeros((self.nj, self.n)), np.tile(np.array(qj0).reshape(-1,1), (1,self.n+1)), np.zeros((self.nj, self.n+1))

    def gravity_compensation(self, p0, R0, qj0):
        """
        Direct, non-optimized feedforward: h(q, 0, 0) is just the bias force
        (gravity + centrifugal/Coriolis, which is ~0 at zero velocity) at the
        CURRENT measured state. No solve, no warm-start dependence, can't go
        stale. Meant to be summed with tau0, not replace it -- this covers
        "don't sink," the MPC's job is still the rest (stepping, balance).
        """
        w_H_b = np.eye(4)
        w_H_b[0:3, 0:3] = np.array(R0, dtype=float).reshape(3, 3)
        w_H_b[0:3, 3] = np.array(p0, dtype=float).reshape(3)
        v_base0 = np.zeros(6)
        v_joints0 = np.zeros(self.nj)
        h = np.array(self.h_fun(w_H_b, np.array(qj0, dtype=float).reshape(self.nj), v_base0, v_joints0)).flatten()
        return h[6:]  # joint-space component only; base 6 DOF aren't actuated

# if __name__ == "__main__":


# if __name__ == "__main__":
#     import os
#
#     LEG_NAMES = ['FL', 'FR', 'BL', 'BR']
#     JOINT_NAMES = {
#         'FL': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
#         'FR': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
#         'BL': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
#         'BR': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint'],
#     }
#     joints_name_list = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
#
#     urdf_path = os.path.join(os.path.dirname(__file__), '../urdf/quadruped.urdf')
#     mpc = WholeBodyMPC(urdf_path, joints_name_list, n=10, dt=0.02)
#
#     idle_angles = [0.00, 0.45, -0.90]
#     qj0 = np.tile(idle_angles, 4)
#
#     p0 = np.array([0.0, 0.0, 0.0401])
#     R0 = np.eye(3)
#     v0 = np.zeros(mpc.nv)
#     qj_ref = np.tile(qj0.reshape(-1, 1), (1, mpc.n + 1))
#     v_ref = np.zeros((mpc.nj, mpc.n + 1))
#     stance = np.ones((4, mpc.n))
#
#     print("Building and solving one WholeBodyMPC step...")
#     tau0, qj_1, vj_1 = mpc.solve(p0, R0, qj0, v0, qj_ref, v_ref, stance)
#     print(f"tau0: {tau0}")
#     Fc0 = np.array(mpc._last_sol.value(mpc._Fc[:, 0])).reshape(4, 3)
#     a0 = np.array(mpc._last_sol.value(mpc._a[:, 0])).flatten()
#     print(f"Fc per foot (node 0): {Fc0}")
#     print(f"Total vertical force: {Fc0[:,2].sum():.2f} N  (expect roughly 12.2 * 9.81 = {12.2*9.81:.1f} N)")
#     print(f"Base + joint acceleration (node 0): {a0}")
#     print("y-acceleration across the horizon:", 
#       [round(float(mpc._last_sol.value(mpc._a[1, k])), 4) for k in range(mpc.n)])
