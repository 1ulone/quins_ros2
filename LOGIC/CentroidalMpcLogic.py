import math as m
import time
import numpy as np
import casadi as cs
import pinocchio as pin

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")         # small matrices: one BLAS thread is faster than many fighting
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # must be set before numpy is imported

GRAVITY = 9.81
N_STATE = 13
N_FORCE = 12

N_NODES = 10
NODE_DT = 0.025
DEFAULT_WEIGHTS = {
    "com_pos": (0.0, 0.0, 500.0),
    "rpy":     (0.2, 0.2, 0.0),
    "com_vel": (20.0, 5.0, 0.0),
    "omega":   (0.2, 0.2, 1.0),
    "force":   1.0e-6,
}


def skew(r):
    # r x f written as skew(r) @ f
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])

def yaw_rot(psi):
    c, s = m.cos(psi), m.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

class CentroidalMPC:
    def __init__(
        self,
        urdf_path,
        joints_name_list,
        inertia_pose,
        mu=0.6,
        fz_max=400.0,
        n_nodes=N_NODES,
        node_dt=NODE_DT,
        weights=None,
        qp_solver="qpoases",
    ):
        self.mu = mu
        self.fz_max = fz_max
        self.n = n_nodes
        self.dt = node_dt
        self.qp_solver = qp_solver

        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        q_diag = np.concatenate([w["com_pos"], w["rpy"], w["com_vel"], w["omega"], [0.0]])
        self.q_bar = np.tile(q_diag, n_nodes)     # diagonal of Q over the whole horizon
        self.R_bar = w["force"] * np.eye(N_FORCE * n_nodes)

        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.q_idx = [self.model.joints[self.model.getJointId(j)].idx_q for j in joints_name_list]
        self.v_idx = [self.model.joints[self.model.getJointId(j)].idx_v for j in joints_name_list]
        self.mass = float(pin.computeTotalMass(self.model))
        self.I_body = self._composite_inertia(inertia_pose)
        self.I_body_inv = np.linalg.inv(self.I_body)

        self._build_solver()
        self.U = np.zeros(N_FORCE * n_nodes)
        self.ok = False
        self.build_s = 0.0
        self.solve_s = 0.0

    def _composite_inertia(self, joint_q):
        q_pin = pin.neutral(self.model)
        q_pin[self.q_idx] = joint_q
        pin.ccrba(self.model, self.data, q_pin, np.zeros(self.model.nv))
        return np.array(self.data.Ig.inertia)

    def _build_solver(self):
        nu = N_FORCE * self.n
        rows = []
        for foot_node in range(4 * self.n):
            c = 3 * foot_node
            for sx, sy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                r = np.zeros(nu)
                r[c], r[c + 1], r[c + 2] = sx, sy, -self.mu
                rows.append(r)
        self.A_dm = cs.sparsify(cs.DM(np.array(rows)))
        self.lba = np.full(len(rows), -np.inf)
        self.uba = np.zeros(len(rows))
        if self.qp_solver == "osqp":
            opts = {"osqp": {"verbose": False},       # OSQP default tolerances (eps_abs = eps_rel = 1e-3)
                    "warm_start_primal": True, "error_on_fail": False}
        else:
            opts = {"printLevel": "none", "error_on_fail": False}
        self.qp = cs.conic('centroidal_mpc', self.qp_solver,
                           {'h': cs.Sparsity.dense(nu, nu), 'a': self.A_dm.sparsity()}, opts)

    def com_state(self, p, R, q, v_lin, omega_body, qd):
        q_pin = pin.neutral(self.model)
        q_pin[0:3] = p
        q_pin[3:7] = pin.Quaternion(R).coeffs()
        q_pin[self.q_idx] = q
        v_pin = np.zeros(self.model.nv)
        v_pin[0:3] = R.T @ v_lin
        v_pin[3:6] = omega_body
        v_pin[self.v_idx] = qd
        pin.centerOfMass(self.model, self.data, q_pin, v_pin)
        return self.data.com[0].copy(), self.data.vcom[0].copy()

    @staticmethod
    def state_vector(com_pos, R, yaw, com_vel, omega_world):
        roll = m.atan2(R[2, 1], R[2, 2])
        pitch = -m.asin(float(np.clip(R[2, 0], -1.0, 1.0)))
        return np.concatenate([com_pos, [roll, pitch, yaw], com_vel, omega_world, [1.0]])

    def reference(self, com_pos, yaw, com_z_des, v_cmd_xy, yaw_rate):
        x_ref = np.zeros((self.n, N_STATE))
        for k in range(self.n):
            t_k = (k + 1) * self.dt
            x_ref[k, 0:2] = com_pos[0:2] + v_cmd_xy * t_k
            x_ref[k, 2] = com_z_des
            x_ref[k, 5] = yaw + yaw_rate * t_k
            x_ref[k, 6:8] = v_cmd_xy
            x_ref[k, 11] = yaw_rate
            x_ref[k, 12] = 1.0
        return x_ref

    def _dynamics(self, yaw, r_feet):
        Rz = yaw_rot(yaw)
        I_world_inv = Rz @ self.I_body_inv @ Rz.T
        A = np.zeros((N_STATE, N_STATE))
        A[0:3, 6:9] = np.eye(3)
        A[3:6, 9:12] = Rz.T
        A[8, 12] = -GRAVITY
        B = np.zeros((N_STATE, N_FORCE))
        for i in range(4):
            B[6:9, 3 * i: 3 * i + 3] = np.eye(3) / self.mass
            B[9:12, 3 * i: 3 * i + 3] = I_world_inv @ skew(r_feet[i])
        return A, B

    def solve(self, x0, x_ref, r_feet, stance):
        """x0: (13,) measured state. x_ref: (n, 13) reference for nodes 1..n.
        r_feet: (n, 4, 3) foot positions from the CoM per node 0..n-1. stance: (n, 4), 1 = stance.
        Returns the node-0 forces (12,) and the CoM linear / angular accelerations they cause."""
        t_start = time.perf_counter()
        n = self.n

        Sx = np.zeros((N_STATE * n, N_STATE))
        Su = np.zeros((N_STATE * n, N_FORCE * n))
        row_x = np.eye(N_STATE)
        row_u = np.zeros((N_STATE, N_FORCE * n))
        for k in range(n):
            yaw_lin = x0[5] if k == 0 else x_ref[k - 1, 5]
            A, B = self._dynamics(yaw_lin, r_feet[k])
            Ad = np.eye(N_STATE) + A * self.dt
            Bd = B * self.dt
            row_x = Ad @ row_x
            row_u = Ad @ row_u
            row_u[:, N_FORCE * k: N_FORCE * (k + 1)] += Bd
            Sx[N_STATE * k: N_STATE * (k + 1)] = row_x
            Su[N_STATE * k: N_STATE * (k + 1)] = row_u

        QSu = self.q_bar[:, None] * Su
        H = 2.0 * (Su.T @ QSu + self.R_bar)
        g = 2.0 * QSu.T @ (Sx @ x0 - x_ref.reshape(-1))
        t_built = time.perf_counter()

        flag = np.asarray(stance, dtype=float).reshape(-1)
        lim_xy = self.mu * self.fz_max
        lbx, ubx = np.zeros(N_FORCE * n), np.zeros(N_FORCE * n)
        lbx[0::3], ubx[0::3] = -lim_xy * flag, lim_xy * flag
        lbx[1::3], ubx[1::3] = -lim_xy * flag, lim_xy * flag
        ubx[2::3] = self.fz_max * flag

        res = self.qp(h=H, g=g, a=self.A_dm, lba=self.lba, uba=self.uba, lbx=lbx, ubx=ubx, x0=self.U)
        self.ok = bool(self.qp.stats()["success"])
        if self.ok:
            self.U = np.array(res["x"]).flatten()
        self.build_s = t_built - t_start
        self.solve_s = time.perf_counter() - t_built

        f0 = self.U[0:N_FORCE]
        acc_lin = f0.reshape(4, 3).sum(axis=0) / self.mass - np.array([0.0, 0.0, GRAVITY])
        Rz = yaw_rot(x0[5])
        torque = sum(np.cross(r_feet[0][i], f0[3 * i: 3 * i + 3]) for i in range(4))
        acc_ang = Rz @ self.I_body_inv @ Rz.T @ torque
        return f0, acc_lin, acc_ang

    def acceleration(self, forces, r_feet_now, yaw):
        acc_lin = forces.reshape(4, 3).sum(axis=0) / self.mass - np.array([0.0, 0.0, GRAVITY])
        Rz = yaw_rot(yaw)
        torque = sum(np.cross(r_feet_now[i], forces[3 * i: 3 * i + 3]) for i in range(4))
        return acc_lin, Rz @ self.I_body_inv @ Rz.T @ torque

def _mpc_worker(conn, urdf_path, joints_name_list, inertia_pose, kwargs):
    # separate process with its own solver; always answers the newest request only
    mpc = CentroidalMPC(urdf_path, joints_name_list, inertia_pose, **kwargs)
    conn.send("ready")
    while True:
        msg = conn.recv()
        while conn.poll():
            msg = conn.recv()
        if msg is None:
            break
        t_req, x0, x_ref, r_feet, stance = msg
        mpc.solve(x0, x_ref, r_feet, stance)
        conn.send((t_req, mpc.U.reshape(mpc.n, 4, 3).copy(), mpc.ok, mpc.build_s + mpc.solve_s))
    conn.close()

class MpcClient:
    """Runs the MPC in its own process at the node rate; the control loop never waits for it.
    Between solves, the force for 'now' is read from the newest plan by the time elapsed since its request."""

    def __init__(self, urdf_path, joints_name_list, inertia_pose, **kwargs):
        import multiprocessing as mp
        self.model = CentroidalMPC(urdf_path, joints_name_list, inertia_pose, **kwargs)   # CoM, inertia, reference only
        self.n, self.dt = self.model.n, self.model.dt
        ctx = mp.get_context("fork")
        self.conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(target=_mpc_worker, daemon=True,
                                args=(child_conn, urdf_path, joints_name_list, inertia_pose, kwargs))
        self.proc.start()
        assert self.conn.recv() == "ready"
        self.busy = False
        self.next_req_t = 0.0
        self.plan = None            # (request time, forces (n, 4, 3))
        self.stats = {"sent": 0, "solved": 0, "failed": 0, "skipped": 0, "stale": 0, "solve_s": []}

    def due(self, t):
        # True once per node period; a period where the worker is still busy is skipped and counted
        if t + 1e-9 < self.next_req_t:
            return False
        while self.next_req_t <= t + 1e-9:
            self.next_req_t += self.dt
        if self.busy:
            self.stats["skipped"] += 1
            return False
        return True

    def send(self, t, x0, x_ref, r_feet, stance):
        self.conn.send((t, x0, x_ref, r_feet, stance))
        self.busy = True
        self.stats["sent"] += 1

    def poll(self):
        while self.conn.poll():
            t_req, forces, ok, solve_s = self.conn.recv()
            self.busy = False
            self.stats["solve_s"].append(solve_s)
            if ok:
                self.plan = (t_req, forces)
                self.stats["solved"] += 1
            else:
                self.stats["failed"] += 1

    def force_now(self, t):
        # None until the first plan arrives
        if self.plan is None:
            return None
        t_req, forces = self.plan
        k = int((t - t_req) / self.dt)
        if k >= self.n:
            self.stats["stale"] += 1
            k = self.n - 1
        return forces[max(k, 0)].reshape(-1).copy()

    def summary(self):
        s = self.stats
        ms = 1e3 * np.array(s["solve_s"]) if s["solve_s"] else np.zeros(1)
        print("\n===== MPC =====")
        print(f"requests           : sent {s['sent']}   solved {s['solved']}   failed {s['failed']}   skipped (busy) {s['skipped']}")
        print(f"stale plan reads   : {s['stale']} control ticks")
        print(f"solve time         : mean {ms.mean():.2f}   p99 {np.percentile(ms, 99):.2f}   max {ms.max():.2f} ms "
              f"(node period {1e3 * self.dt:.0f} ms)", flush=True)

    def close(self):
        try:
            self.conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(timeout=1.0)

if __name__ == "__main__":
    # Standalone check: no simulation. 1) standing force split, 2) solve time over a trot horizon
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from LOGIC.GaitLogic import LEG_NAMES, JOINT_NAMES
    from LOGIC.GaitPlanner import GaitPlanner, GAIT_TABLE

    NOMINAL_STANCE = np.tile([0.0, -0.83, 1.113], 4)
    FOOT_FRAMES = ('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT')
    V_CMD = np.array([0.38, 0.0])       # current WALK speed (step_len * freq / duty)
    N_TRIALS = 2000

    joints = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
    qp_solver = sys.argv[1] if len(sys.argv) > 1 else "qpoases"
    mpc = CentroidalMPC(str(root / "urdf" / "quadruped.urdf"), joints, NOMINAL_STANCE, qp_solver=qp_solver)
    print(f"[mpc] qp solver = {qp_solver}")
    print(f"[mpc] mass = {mpc.mass:.2f} kg, composite inertia diag = {np.round(np.diag(mpc.I_body), 4)} kg m^2")

    # robot standing in NOMINAL_STANCE, lowest foot at z = 0
    q_pin = pin.neutral(mpc.model)
    q_pin[mpc.q_idx] = NOMINAL_STANCE
    pin.framesForwardKinematics(mpc.model, mpc.data, q_pin)
    feet = np.array([mpc.data.oMf[mpc.model.getFrameId(f)].translation for f in FOOT_FRAMES])
    p = np.array([0.0, 0.0, -feet[:, 2].min()])
    feet_world = feet + p
    R = np.eye(3)
    com, com_vel = mpc.com_state(p, R, NOMINAL_STANCE, np.zeros(3), np.zeros(3), np.zeros(12))
    r_feet = np.tile(feet_world - com, (mpc.n, 1, 1))

    # 1) all four feet down, no command: forces must carry the weight, CoM acceleration near zero
    x0 = mpc.state_vector(com, R, 0.0, com_vel, np.zeros(3))
    x_ref = mpc.reference(com, 0.0, com[2], np.zeros(2), 0.0)
    f0, acc_lin, acc_ang = mpc.solve(x0, x_ref, r_feet, np.ones((mpc.n, 4)))
    print(f"[stand] ok = {mpc.ok}, fz per foot = {np.round(f0[2::3], 2)} N, sum = {f0[2::3].sum():.2f} N "
          f"(m g = {mpc.mass * GRAVITY:.2f} N), acc_lin = {np.round(acc_lin, 3)}")

    # 2) trot horizon from the real GaitPlanner, random phase and small state noise
    planner = GaitPlanner(GAIT_TABLE["WALK"], {leg: feet_world[i, 0:2] - p[0:2] for i, leg in enumerate(LEG_NAMES)})
    rng = np.random.default_rng(0)
    times, fails = [], 0
    for trial in range(N_TRIALS):
        t0 = rng.uniform(0.0, 1.0)
        stance = np.array(planner.horizon(t0, mpc.n, mpc.dt, mpc.mass)["stance"], dtype=float)
        com_noisy = com + rng.normal(0.0, 0.005, 3)
        vel_noisy = np.array([V_CMD[0], V_CMD[1], 0.0]) + rng.normal(0.0, 0.05, 3)
        x0 = mpc.state_vector(com_noisy, R, 0.0, vel_noisy, rng.normal(0.0, 0.1, 3))
        x_ref = mpc.reference(com_noisy, 0.0, com[2], V_CMD, 0.0)
        mpc.solve(x0, x_ref, r_feet, stance)
        fails += int(not mpc.ok)
        if trial >= 10:                     # skip the first solves (memory warm-up)
            times.append((mpc.build_s, mpc.solve_s))

    build_ms, qp_ms = 1e3 * np.array(times).T
    print(f"[trot] solves = {N_TRIALS}, failed = {fails}")
    print(f"[trot] condense (numpy): mean {build_ms.mean():.2f}   p99 {np.percentile(build_ms, 99):.2f} ms")
    print(f"[trot] QP solve        : mean {qp_ms.mean():.2f}   p99 {np.percentile(qp_ms, 99):.2f}   max {qp_ms.max():.2f} ms")
    print(f"[trot] last solver stats: {mpc.qp.stats()}")
    print(f"[trot] free time in the 8 ms tick at the WBC p99 (5.85 ms): {8.0 - 5.85:.2f} ms")
