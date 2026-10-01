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
        n=20,
        dt=0.02,
        mu=0.6,
        foot_names=('tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link'),
        tau_max=100.0,
        fz_max=400.0,
        joint_damping=10.0,
        base_w=(1000.0, 3000.0)
    ):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.tau_max = tau_max
        self.fz_max = fz_max
        self.fc_w = 1e-3
        self.q_joint_w = np.tile([30.0, 1.0, 1.0], 4)
        self.joints_name_list = joints_name_list
        self.nj = len(joints_name_list)
        self.joint_damping = joint_damping
        self.base_z_w, self.base_R_w = base_w
        self.foot_names = list(foot_names)
        self.nv = 6 + self.nj

        self.kindyn = KinDynComputations(urdf_path, joints_name_list)
        self.kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

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
        z_des_p = opti.parameter()
        R_des_p = opti.parameter(3, 3)

        self._qj_w_p = opti.parameter()
        self._vbase_w_p = opti.parameter()
        self._vj_w_p = opti.parameter()
        self._tau_w_p = opti.parameter()
        self._a_w_p = opti.parameter()

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
                # contact_wrench += Ji[0:3, :].T @ Fi
                contact_wrench += stance_p[i, k] * (Ji[0:3, :].T @ Fi)

                if k > 0:
                    st = stance_p[i, k]
                    foot_vel = Ji[0:3, :] @ v[k]
                    sub(f"noslip_{legs[i]}", k, opti.bounded(-no_slip_M * (1 - st), foot_vel, no_slip_M * (1 - st)))

                    swing_slack = no_slip_M * (1 - vz_active_p[i, k] * (1 - st))
                    sub(f"swingvz_{legs[i]}", k, opti.bounded(-swing_slack, foot_vel[2] - swing_vz_p[i, k], swing_slack))

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

            # ---- Stage cost ----
            cost += self._qj_w_p * cs.sumsqr(cs.sqrt(self.q_joint_w) * (qj[k] - q_nom_p))
            cost += self._vbase_w_p * cs.sumsqr(v_base_k - v_base_des_p)
            cost += self._vj_w_p * cs.sumsqr(v_joints_k)
            cost += self._tau_w_p * cs.sumsqr(tau_j[k])
            cost += self._a_w_p * cs.sumsqr(a[k])
            cost += self.fc_w * cs.sumsqr(Fc[k] - fc_des_p[:, k])
            cost += self.base_z_w * (p[k][2] - z_des_p) ** 2
            cost += self.base_R_w * cs.sumsqr(R[k] - R_des_p)

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
            "fatrop.max_iter": 100,           # now actually applied; solves finish in about 20 to 30
            "fatrop.tol": 1e-3,
            "fatrop.mu_init": 0.1,
        }
        opti.solver("fatrop", self._solver_opts)
        self._struct_sig = None

        self.opti = opti
        # Horizon-shaped expressions for reading results (same shapes as before)
        self._p = cs.horzcat(*p)          # 3 x (n+1)
        self._R = R                       # list of n+1 (3x3)
        self._qj = cs.horzcat(*qj)        # nj x (n+1)
        self._v = cs.horzcat(*v)          # nv x (n+1)
        self._a = cs.horzcat(*a)          # nv x n
        self._tau_j = cs.horzcat(*tau_j)  # nj x n
        self._Fc = cs.horzcat(*Fc)        # 12 x n
        self._Fc_list = Fc                # per-stage variables, needed for set_initial
        self._z_des_p, self._R_des_p = z_des_p, R_des_p
        self._p0_p, self._R0_p, self._qj0_p, self._v0_p = p0_p, R0_p, qj0_p, v0_p
        self._q_nom_p, self._v_base_des_p, self._swing_vz_p, self._stance_p = (
            q_nom_p, v_base_des_p, swing_vz_p, stance_p
        )
        self._vz_active_p = vz_active_p
        self._fc_des_p = fc_des_p

    def constraint_report(self, top=8):
        """DIAGNOSTIC: rank constraint groups at the last iterate by |multiplier| and by violation."""
        try:
            dbg = self.opti.debug
            g = np.array(dbg.value(self.opti.g)).flatten()
            lam = np.array(dbg.value(self.opti.lam_g)).flatten()
            lbg = np.array(dbg.value(self.opti.lbg)).flatten()
            ubg = np.array(dbg.value(self.opti.ubg)).flatten()
        except Exception as e:
            print(f"  [con report] unavailable: {e!r}", flush=True)
            return
        viol = np.maximum(np.maximum(lbg - g, g - ubg), 0.0)
        rows = []
        for name, k, s, e in self._con_groups:
            if e <= s:
                continue
            rows.append((name, k, float(np.max(np.abs(lam[s:e]))), float(np.max(viol[s:e])),
                         float(np.min(ubg[s:e] - lbg[s:e]))))
        print("  [con report] largest |multiplier| (name, node, max|lam|, max viol, min width):", flush=True)
        for r in sorted(rows, key=lambda r: -r[2])[:top]:
            print(f"    {r[0]:<14} k={r[1]:<2} lam={r[2]:.2e} viol={r[3]:.2e} width={r[4]:.1e}", flush=True)
        print("  [con report] largest violation:", flush=True)
        for r in sorted(rows, key=lambda r: -r[3])[:top]:
            print(f"    {r[0]:<14} k={r[1]:<2} lam={r[2]:.2e} viol={r[3]:.2e} width={r[4]:.1e}", flush=True)

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
        mpc_weights=(100.0, 25.0, 0.04, 0.0000004, 0.0004),
        vz_active_schedule=None,
        fc_guess=None,
        z_des=None,
        R_des=None,
    ):
        self.last_solve_fresh = False 
        self.last_solve_accepted = False 

        if hasattr(self, "_solve_thread") and self._solve_thread.is_alive():
            if hasattr(self, "_last"):
                return self._last
            return (np.zeros((self.nj, self.n)),
                    np.tile(np.array(qj0, dtype=float).reshape(-1, 1), (1, self.n + 1)),
                    np.zeros((self.nj, self.n + 1)))


        st_arr = np.asarray(stance_schedule, dtype=float)
        va_arr = np.ones((4, self.n)) if vz_active_schedule is None else np.asarray(vz_active_schedule, dtype=float)
        sig = tuple((int(st_arr[:, k].sum()), int(((1 - st_arr[:, k]) * va_arr[:, k]).sum())) for k in range(self.n))
        if self._struct_sig is not None and sig != self._struct_sig:
            self.opti.solver("fatrop", self._solver_opts)
        self._struct_sig = sig
        self.opti.set_value(self._p0_p, p0)
        self.opti.set_value(self._R0_p, R0)
        self.opti.set_value(self._qj0_p, qj0)
        self.opti.set_value(self._v0_p, v0)
        self.opti.set_value(self._q_nom_p, q_nom)
        self.opti.set_value(self._v_base_des_p, v_base_des)
        self.opti.set_value(self._swing_vz_p, swing_vz_schedule)
        self.opti.set_value(self._stance_p, stance_schedule)
        self.opti.set_value(self._vz_active_p, np.ones((4, self.n)) if vz_active_schedule is None else vz_active_schedule)
        self.opti.set_value(self._z_des_p, float(p0[2]) if z_des is None else z_des)
        self.opti.set_value(self._R_des_p, np.eye(3) if R_des is None else R_des)

        self.opti.set_value(self._qj_w_p, mpc_weights[0])
        self.opti.set_value(self._vbase_w_p, mpc_weights[1])
        self.opti.set_value(self._vj_w_p, mpc_weights[2])
        self.opti.set_value(self._tau_w_p, mpc_weights[3])
        self.opti.set_value(self._a_w_p, mpc_weights[4])
        qj0_arr = np.array(qj0, dtype=float)
        v0_arr = np.array(v0, dtype=float)
        p0_arr = np.array(p0, dtype=float)

        sane = (
            np.all(np.isfinite(qj0_arr)) and np.all(np.isfinite(v0_arr)) and np.all(np.isfinite(p0_arr))
            and -3.00 < p0_arr[2] < 3.00
            and np.max(np.abs(v0_arr)) < 50.0
        )
        if not sane:
            print(f"[WholeBodyMPC] refusing solve: state outside sane envelope "
                  f"(base z={p0_arr[2]:.3f}, max|v0|={np.max(np.abs(v0_arr)):.1f})", flush=True)
            safe_qj = np.tile(np.array(q_nom, dtype=float).reshape(-1, 1), (1, self.n + 1))
            self.last_solve_fresh = True
            return np.zeros((self.nj, self.n)), safe_qj, np.zeros((self.nj, self.n + 1))

        if hasattr(self, "_warm"):
            try:
                self.opti.set_initial(self._warm)
            except Exception:
                pass

        # if fc_guess is not None:
        #     fc_guess = np.array(fc_guess, dtype=float).reshape(12, self.n)
        #     for k in range(self.n):
        #         self.opti.set_initial(self._Fc_list[k], fc_guess[:, k])
        fc_des = np.zeros((12, self.n)) if fc_guess is None else np.array(fc_guess, dtype=float).reshape(12, self.n)
        self.opti.set_value(self._fc_des_p, fc_des)   # eq. 3 target for the contact forces
        for k in range(self.n):
            self.opti.set_initial(self._Fc_list[k], fc_des[:, k])

        import threading
        result = {}
        def _run_solve():
            try:
                result["sol"] = self.opti.solve()
            except Exception as e:
                result["exc"] = e
        self._solve_thread = threading.Thread(target=_run_solve, daemon=True)
        self._solve_thread.start()
        first_solve = not getattr(self, "_jit_ready", False)
        self._solve_thread.join(timeout=None if first_solve else 60.0)
        if not self._solve_thread.is_alive():
            self._jit_ready = True
        if self._solve_thread.is_alive():
            print("[WholeBodyMPC] solve TIMED OUT (>60s), abandoning this attempt", flush=True)
            print(f"  [hang inputs] p0={np.round(p0_arr, 4)}  max|v0|={np.max(np.abs(v0_arr)):.3f}\n"
                  f"  qj0={np.round(qj0_arr, 3)}\n"
                  f"  v0={np.round(v0_arr, 3)}\n"
                  f"  v_base_des={np.round(np.array(v_base_des, dtype=float), 3)}\n"
                  f"  stance (rows FL,FR,BL,BR x nodes)=\n{np.array(stance_schedule)}\n"
                  f"  swing_vz=\n{np.round(np.array(swing_vz_schedule, dtype=float), 3)}\n"
                  f"  vz_active=\n{np.array(vz_active_schedule) if vz_active_schedule is not None else 'None'}",
                  flush=True)
            if hasattr(self, "_last"):
                return self._last
            return np.zeros((self.nj, self.n)), np.tile(np.array(qj0).reshape(-1,1), (1,self.n+1)), np.zeros((self.nj, self.n+1))
        if "exc" in result:
            # Solver did not certify convergence. If its last iterate still satisfies every
            # constraint, use it (standard real-time MPC practice); otherwise report and fail.
            dbg = self.opti.debug
            try:
                g = np.array(dbg.value(self.opti.g)).flatten()
                lbg = np.array(dbg.value(self.opti.lbg)).flatten()
                ubg = np.array(dbg.value(self.opti.ubg)).flatten()
                max_viol = float(np.max(np.maximum(np.maximum(lbg - g, g - ubg), 0.0)))
            except Exception:
                max_viol = float("inf")
            if max_viol < 1e-4:
                tau_traj = np.array(dbg.value(self._tau_j)).reshape(self.nj, self.n)
                qj_traj = np.array(dbg.value(self._qj)).reshape(self.nj, self.n + 1)
                v_traj = np.array(dbg.value(self._v[6:, :])).reshape(self.nj, self.n + 1)
                self._warm = dbg.value_variables()
                self._last = (tau_traj, qj_traj, v_traj)
                self.last_solve_fresh = True
                self.last_solve_accepted = True
                if not getattr(self, "_diag_done", False):   # DIAGNOSTIC: first non-converged solve only
                    self._diag_done = True
                    print(f"[diag] full stats: {self.opti.stats()}", flush=True)
                    self.constraint_report(top=10)
                return tau_traj, qj_traj, v_traj
            self.constraint_report()
            raise result["exc"]
        try:
            sol = result["sol"]
            self._last_sol = sol
            self._warm = sol.value_variables()
            tau_traj = np.array(sol.value(self._tau_j)).reshape(self.nj, self.n)
            qj_traj = np.array(sol.value(self._qj)).reshape(self.nj, self.n + 1)
            v_traj = np.array(sol.value(self._v[6:, :])).reshape(self.nj, self.n + 1)
            self._last = (tau_traj, qj_traj, v_traj)
            self.last_solve_fresh = True
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
