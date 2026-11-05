import time
import numpy as np
import casadi as cs
from LOGIC.CentroidalMpcLogic import CentroidalMPC, NODE_DT, GRAVITY, N_FORCE, skew, yaw_rot

POLICY_DT = NODE_DT
N_ACTION = 6
ACC_SCALE = np.array([4.0, 4.0, 15.0, 10.0, 15.0, 5.0])
FORCE_REG = 1.0e-6

OBS_LAYOUT = (
    ("body_pos_err", 3),
    ("body_rot_err", 3),
    ("body_vel_err", 3),
    ("body_angvel_err", 3),
    ("gravity_body", 3),
    ("v_cmd_xy", 2),
    ("yaw_rate_cmd", 1),
    ("feet_from_com", 12),
    ("schedule", 4),
    ("stance", 4), 
    ("phase_sin_cos", 2), 
    ("prev_action", N_ACTION), 
)
N_OBS = sum(n for _, n in OBS_LAYOUT)

class ForceDistributor:
    # a quick one time instance that uses the same logic as mpc
    # without it horizon.

    def __init__(self, mass, I_body, mu, fz_max, reg=FORCE_REG):
        self.mass = mass
        self.I_body = I_body
        self.mu = mu
        self.fz_max = fz_max
        self.reg = reg
        
        rows = []
        for foot in range(4):
            c = 3 * foot
            for sx, sy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                r = np.zeros(N_FORCE)
                r[c], r[c + 1], r[c + 2] = sx, sy, -mu
                rows.append(r)

        self.A_fric = cs.sparsify(cs.DM(np.array(rows)))
        self.lba = np.full(len(rows), -np.inf)
        self.uba = np.zeros(len(rows))
        self.qp = cs.conic(
            'force_distribution',
            'qpoases',
            {'h': cs.Sparsity.dense(N_FORCE, N_FORCE), 'a': self.A_fric.sparsity()},
            {"printLevel": "none", "error_on_fail": False}
        )

        self.ok = False
        self.forces = np.zeros(N_FORCE)

    def solve(self, acc_lin, acc_ang, r_feet, stance, yaw):
        Rz = yaw_rot(yaw)
        I_world = Rz @ self.I_body @ Rz.T
        wrench_map = np.zeros((6, N_FORCE))

        for i in range(4):
            wrench_map[0:3, 3 * i: 3 * i + 3] = np.eye(3)
            wrench_map[3:6, 3 * i: 3 * i + 3] = skew(r_feet[i])

        wrench_wanted = np.concatenate([self.mass * (np.asarray(acc_lin) + np.array([0.0, 0.0, GRAVITY])), I_world @ np.asarray(acc_ang)])

        H = 2.0 * (wrench_map.T @ wrench_map + self.reg * np.eye(N_FORCE))
        g = -2.0 * wrench_map.T @ wrench_wanted

        flag = np.asarray(stance, dtype=float)
        lim_xy = self.mu * self.fz_max
        lbx = np.zeros(N_FORCE)
        ubx = np.zeros(N_FORCE)

        lbx[0::3] = -lim_xy * flag
        ubx[0::3] = lim_xy * flag

        lbx[1::3] = -lim_xy * flag
        ubx[1::3] = lim_xy * flag

        ubx[2::3] = self.fz_max * flag

        res = self.qp(h=H, g=g, a=self.A_fric, lba=self.lba, uba=self.uba, lbx=lbx, ubx=ubx)
        self.ok = bool(self.qp.stats()["success"])
        if self.ok:
            self.forces = np.array(res["x"]).flatten()

        return self.forces.copy()


class RLTopClient:
    # tune from body acceleration into stance-foot forces then taskcontroller
    # feeds wbc like how a mpc would, except this aint using no horizon stuff.

    def __init__(self, urdf_path, joints_name_list, inertia_pose, policy=None, mu=0.6, fz_max=400.0, dt=POLICY_DT):
        self.model = CentroidalMPC(urdf_path, joints_name_list, inertia_pose, mu=mu, fz_max=fz_max)
        self.dist = ForceDistributor(self.model.mass, self.model.I_body, mu, fz_max)
        self.policy = policy
        self.dt = dt
        self.next_t = 0.0
        self.action = np.zeros(N_ACTION)
        self.queued_action = None
        self.obs = None
        self.stats = {
            "decisions": 0,
            "policy_s" : [],
            "force_s" : [],
            "force_failed": 0,
        }

    def due(self, t):
        if t + 1e-9 < self.next_t:
            return False
        while self.next_t <= t + 1e-9:
            self.next_t += self.dt
        return True

    def decide(self):
        if self.policy is not None and self.obs is not None:
            t0 = time.perf_counter()
            action = self.policy(self.obs)
            self.stats["policy_s"].append(time.perf_counter() - t0)
            self.action = np.clip(np.asarray(action, dtype=float).reshape(N_ACTION), -1.0, 1.0)
        elif self.queued_action is not None:
            self.action = np.clip(np.asarray(self.queued_action, dtype=float).reshape(N_ACTION), -1.0, 1.0)
            self.queued_action = None
        self.stats["decisions"] += 1

    def force_now(self, r_feet, stance, yaw_heading, yaw_meas):
        t0 = time.perf_counter()
        R_heading = yaw_rot(yaw_heading)
        acc = ACC_SCALE * self.action
        forces = self.dist.solve(R_heading @ acc[0:3], R_heading @ acc[3:6], r_feet, stance, yaw_meas)
        self.stats["force_s"].append(time.perf_counter() - t0)
        self.stats["force_failed"] += int(not self.dist.ok)
        return forces

    def observe(self, e, e_dot, R, yaw_heading, v_cmd, yaw_rate, r_feet, schedule, stance, phase):
        to_heading = yaw_rot(yaw_heading).T
        self.obs = np.concatenate([
            to_heading @ e[0:3],
            to_heading @ e[3:6],
            to_heading @ e_dot[0:3],
            to_heading @ e_dot[3:6],
            R.T @ np.array([0.0, 0.0, -1.0]),
            to_heading[0:2, 0:2] @ v_cmd,
            [yaw_rate],
            (np.asarray(r_feet) @ to_heading.T).reshape(-1),
            schedule,
            stance,
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
            self.action,
        ]).astype(np.float32)

    def summary(self):
        s = self.stats
        force_ms = 1e3 * np.array(s["force_s"]) if s["force_s"] else np.zeros(1)
        print("\n===== RL TOP LAYER =====")
        print(f"decisions          : {s['decisions']} (every {1e3 * self.dt:.0f} ms)")
        print(f"force QP           : mean {force_ms.mean():.3f}   p99 {np.percentile(force_ms, 99):.3f} ms   failed {s['force_failed']}")
        if s["policy_s"]:
            pol_ms = 1e3 * np.array(s["policy_s"])
            print(f"policy inference   : mean {pol_ms.mean():.3f}   p99 {np.percentile(pol_ms, 99):.3f} ms", flush=True)
