import time
import math as m
from pathlib import Path
import numpy as np

from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.WbcLogic import WholeBodyController
from LOGIC.TaskLogic import TaskController

NOMINAL_STANCE = np.tile([0.0, -0.83, 1.113], 4)
USE_FOSMC = True
USE_RL_TUNER = True
RL_DT = 0.02
RL_TEST = None
WALK_DROP = 0.085
IDLE_DROP = 0.05
GROUND_TAU = 0.2
FOSMC_MEMORY_S = 0.2
DEBOUNCE_S = 0.006
SAFE_TILT = 0.82
ROOT = Path(__file__).resolve().parent.parent
TUNER_LOAD = str(ROOT / "RL" / "tuner_pretrained.npz")   # None = start from scratch
GROUP_ROWS = {"lin": slice(0, 3), "ang": slice(3, 6), "swing": slice(6, 18)}


# quaternion to rotation matrix.
def quat_to_rot(quat):
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

class ControllerCore:
    def __init__(self, cfg):
        self.ctrl_dt = cfg["ctrl_dt"]
        self.z_nom = cfg["z_nom"]
        joints = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]

        self.logic = GaitLogic()
        self.logic.robot_mass = cfg["robot_mass"]
        self.logic_dt = 1.0 / self.logic.control_rate
        self.wbc = WholeBodyController(str(ROOT / "urdf" / "quadruped.urdf"), joints, joint_damping=cfg["joint_damping"], ctrl_dt=self.ctrl_dt)

        self.task = TaskController(
            self.logic,
            NOMINAL_STANCE,
            cfg["z_nom"],
            np.asarray(cfg["foot_nom_xy"], dtype=float),
            use_fosmc=USE_FOSMC,
            fosmc_dt=self.ctrl_dt,
            walk_drop=WALK_DROP,
            idle_drop=IDLE_DROP,
            foot_r=cfg["foot_r"],
            ground_tau=GROUND_TAU,
            fosmc_memory=max(1, round(FOSMC_MEMORY_S / self.ctrl_dt)),
            contact_debounce=max(1, round(DEBOUNCE_S / self.ctrl_dt)),
        )

        self.tuner = None
        if USE_RL_TUNER and self.task.fosmc is not None:
            from RL.MainTuner import RLTuner
            self.tuner = RLTuner(self.task.fosmc, test=RL_TEST)

            if TUNER_LOAD and RL_TEST is None:
                self.tuner.load(TUNER_LOAD)
                self.tuner.actor.explore *= 0.5 # type: ignore


        self.t_prev = None
        self.next_logic_t = 0.0
        self.next_rl_t = 0.0
        self.yaw = 0.0
        self.prev_raw_yaw = 0.0
        self.walk_t0 = None
        self.fell = None
        self.req_id = None
        self.part_s = np.zeros(5)

    def step(self, t, st):
        marks = [time.perf_counter()]
        lg, task, wbc = self.logic, self.task, self.wbc
        p, q, qd = st["p"], st["q"], st["qd"]
        R = quat_to_rot(st["quat"])
        v_base = np.concatenate([st["v_lin"], R @ st["omega"]])
        
        n_steps = 1 if self.t_prev is None else max(1, round((t - self.t_prev) / self.ctrl_dt))
        dt = n_steps * self.ctrl_dt
        self.t_prev = t

        # input control, read input
        if st["req_id"] != self.req_id:
            self.req_id = st["req_id"]
            if st["req"] is not None:
                lg.request_state(st["req"])

        # gait logic on its own rate, on the sim clock
        if t + 1e-9 >= self.next_logic_t:
            while self.next_logic_t <= t + 1e-9:
                self.next_logic_t += self.logic_dt
            raw_yaw = m.atan2(R[1, 0], R[0, 0])
            self.yaw += (raw_yaw - self.prev_raw_yaw + m.pi) % (2.0 * m.pi) - m.pi
            self.prev_raw_yaw = raw_yaw

            lg.current_yaw = self.yaw
            lg.loop_step(t)

            if self.walk_t0 is None and (lg.walking or lg.turning):
                self.walk_t0 = t


        # wbc 
        marks.append(time.perf_counter())
        wbc.update_state(p=p, R=R, q=q, v_base=v_base, v_joints=qd)
        marks.append(time.perf_counter())
        cmd = task.compute(
            t=t,
            dt=dt,
            p=p,
            R=R,
            v_base=v_base,
            q=q,
            v_joints=qd,
            pf=wbc.pf,
            vf=wbc.vf,
            fz=st["fz"],
            acc_achieved=np.concatenate([wbc.last["acc_base"], wbc.last["acc_foot"]]),
        )
        marks.append(time.perf_counter())
        tau = wbc.solve(**cmd)["tau"]
        marks.append(time.perf_counter())
        dbg = task.debug

        # fall check 
        tilted = R[2, 2] < SAFE_TILT
        jumped = st["v_lin"][2] > 1.0
        too_low = p[2] - dbg["ground_z"] < 0.5 * self.z_nom
        if self.fell is None and (tilted or jumped or too_low):
            self.fell = t
            print(f"[ctrl] fall check tripped at t = {t:.3f} s: tilt R22 = {R[2, 2]:.3f} (trip < {SAFE_TILT}), "
                  f"vz = {st['v_lin'][2]:+.3f} m/s (trip > 1.0), "
                  f"height above ground = {p[2] - dbg['ground_z']:.3f} m (trip < {0.5 * self.z_nom:.3f})", flush=True)

        # reinforcement learning gain tuner
        if self.tuner is not None:
            record = (self.walk_t0 is not None and t >= self.walk_t0 + 1.0 and self.fell is None and (lg.walking or lg.turning))
            self.tuner.observe(dbg, tau, record=record)

            if t + 1e-9 >= self.next_rl_t:
                while self.next_rl_t <= 1e-9:
                    self.next_rl_t += RL_DT

                self.tuner.step(fell=self.fell is not None)

        marks.append(time.perf_counter())
        self.part_s = np.diff(np.array(marks))     # prep + gait logic, dynamics terms, task targets, QP, fall check + tuner
        return np.array(tau, dtype=float), self._diag(dbg), lg.current_state

    def _diag(self, dbg):
        lg, task = self.logic, self.task

        def group_mean(x):
            out = []
            for rows in GROUP_ROWS.values():
                act = dbg["active"][rows]
                out.append(float(np.mean(np.abs(x[rows][act]))) if np.any(act) else 0.0)
            return out

        s_group, miss_group = group_mean(dbg["s"]), group_mean(dbg["miss"])
        diag = {
            "walk_active": float(lg.walking or lg.turning),
            "v_cmd_x": float(task.v_cmd[0]),
            "v_cmd_y": float(task.v_cmd[1]),
            "wz_cmd": float(lg.yaw_rate) if lg.turning else 0.0,
            "z_des": float(task.z_des),
            "ground_z": float(dbg["ground_z"]),
            "fl_z_des": float(dbg["pf_d"][2]),
            "s_max": float(dbg["s_max"]),
            "frozen": float(dbg["frozen"]),
            "wbc_ok": float(self.wbc.ok),
            "fell_t": -1.0 if self.fell is None else float(self.fell),
            "yaw": float(self.yaw),
            "target_yaw": float(lg.target_yaw),
        }
        for i, leg in enumerate(LEG_NAMES):
            diag[f"st_{leg}"] = float(dbg["stance"][i])
        for j, g in enumerate(GROUP_ROWS):
            diag[f"s_{g}"] = s_group[j]
            diag[f"miss_{g}"] = miss_group[j]
        return diag

    def summary(self):
        if self.tuner is not None:
            self.tuner.summary()
