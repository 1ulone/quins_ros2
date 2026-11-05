import os
import sys
import time
import ctypes
from pathlib import Path
from contextlib import contextmanager

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LOGIC.GaitLogic import LEG_NAMES, JOINT_NAMES
from LOGIC.GaitPlanner import FORWARD
from LOGIC.ControllerCore import ControllerCore, NOMINAL_STANCE, ROOT as CORE_ROOT
from LOGIC.WbcLogic import WholeBodyController
from LOGIC.RLTopLogic import N_OBS, N_ACTION
from LOGIC.CentroidalMpcLogic import DEFAULT_WEIGHTS as MPC_WEIGHTS

@contextmanager
def silence_output():
    """Mute stdout and stderr, from Python and from C/C++ libraries: qpOASES prints its license banner
    and Pinocchio's URDF parser warns about the xacro / mujoco tags every time the controller is rebuilt,
    which is every reset. A Python exception still shows: its traceback prints after the streams are back."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    sys.stderr.flush()
    libc.fflush(None)
    saved_fds = (os.dup(1), os.dup(2))
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        libc.fflush(None)
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(devnull_fd)
        os.close(saved_fds[0])
        os.close(saved_fds[1])

CTRL_DECIM = 4              # physics steps per control period: 8 ms, same as SimNode and the Jetson
EPISODE_S = 20.0
FOOT_NAMES = ('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT')

# reward = alive bonus - the centroidal MPC's stage cost (same Q and R as the MPC, Song et al. 2023 "same objective"),
# every term multiplied by the time it covers. The alive bonus keeps the reward positive so that falling (ending
# the episode) never pays: 1.0 is above the zero-action MPC cost on ~98% of ticks (randomized plant, p99 = 1.29).
# A negative step total is still clipped to 0, legged_gym's only_positive_rewards ("avoids early termination problems").
MPC_STATE_Q = np.concatenate([MPC_WEIGHTS["com_pos"], MPC_WEIGHTS["rpy"], MPC_WEIGHTS["com_vel"], MPC_WEIGHTS["omega"]])
MPC_FORCE_R = MPC_WEIGHTS["force"]
REWARD_SCALES = {
    "alive": 1.0,
    "mpc_state_cost": -1.0,
    "mpc_force_cost": -1.0,
}

# per-episode plant mismatch while training, same ranges as the thesis tuner pretraining (RL/Pretrain.py RANGES)
RAND_RANGES = {"mass": (0.8, 1.2), "friction": (0.5, 1.25), "motor": (0.8, 1.2), "push": (0.0, 0.5)}
PUSH_EVERY = 10.0           # s between pushes while walking, same as SimNode
GAITS = ("WALK", "WALK_BACK", "TURN", "TURN_RIGHT")
TEST_SCRIPT = [(2.0, "WALK"), (12.0, "TURN"), (20.0, "IDLE")]   # the journal walk / turn / idle script


class RlTopEnv(gym.Env):
    """MuJoCo plant + the real ControllerCore (WBC + body PD + RL top layer), stepped in lockstep.
    One env step = one policy decision (POLICY_DT = 25 ms, about 3 control ticks of 8 ms)."""

    metadata = {"render_modes": []}

    def __init__(self, randomize=True, script=None, episode_s=EPISODE_S):
        self.randomize = randomize
        self.fixed_script = script          # None = a random state sequence every episode
        self.episode_s = episode_s

        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'urdf' / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        model = self.model

        joints = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
        self.qpos_idx = model.jnt_qposadr[jids]
        self.qvel_idx = model.jnt_dofadr[jids]
        self.act_idx = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor") for j in joints])
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f) for f in FOOT_NAMES]
        self.foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_collision") for f in FOOT_NAMES]
        lf_hip_x = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'LF_HIP')][0]
        assert np.sign(lf_hip_x) == FORWARD, f"LF_HIP is at x = {lf_hip_x:+.3f} but FORWARD = {FORWARD}"
        joint_damping = model.dof_damping[self.qvel_idx]
        assert np.allclose(joint_damping, joint_damping[0]), "WBC expects one damping value for all joints"

        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        floor_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('floor', 'uneven_floor')]
        self.friction_geoms = [g for g in floor_geoms if g >= 0] + self.foot_geoms
        self.base_mass_nom = float(model.body_mass[self.base_id])
        self.friction_nom = model.geom_friction[self.friction_geoms, 0].copy()

        # start pose and the robot constants the controller needs, measured once on the nominal model
        self._start_pose()
        self.qpos0 = self.data.qpos.copy()
        foot_r = model.geom_size[self.foot_geoms[0]][0]
        z_nom = float(self.data.qpos[2]) - (min(self.data.xpos[f][2] for f in self.foot_ids) - foot_r)
        self.cfg = {
            "ctrl_dt": float(model.opt.timestep) * CTRL_DECIM,
            "z_nom": z_nom,
            "foot_nom_xy": np.array([self.data.xpos[f][0:2] - self.data.qpos[0:2] for f in self.foot_ids]),
            "foot_r": float(foot_r),
            "robot_mass": float(model.body_mass.sum()),
            "joint_damping": float(joint_damping[0]),
        }
        self.ctrl_dt = self.cfg["ctrl_dt"]
        with silence_output():
            self.wbc = WholeBodyController(str(CORE_ROOT / "urdf" / "quadruped.urdf"), joints,joint_damping=self.cfg["joint_damping"], ctrl_dt=self.ctrl_dt)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ACTION,), dtype=np.float32)
        self.core = None

    def _start_pose(self):
        # same as SimNode: NOMINAL_STANCE with the feet just clear of the floor
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        data.qpos[self.qpos_idx] = NOMINAL_STANCE
        mujoco.mj_forward(model, data)
        data.qpos[2] -= min(data.xpos[f][2] for f in self.foot_ids)
        mujoco.mj_forward(model, data)
        while data.ncon > 0:
            data.qpos[2] += 0.01
            mujoco.mj_forward(model, data)

    def _random_script(self):
        rng = self.np_random
        first = str(rng.choice(GAITS))
        script = [(float(rng.uniform(1.0, 2.0)), first)]
        if rng.uniform() < 0.5:
            second = str(rng.choice([g for g in GAITS + ("IDLE",) if g != first]))
            script.append((float(rng.uniform(8.0, 14.0)), second))
        return script

    def _sensors(self):
        model, data = self.model, self.data
        fz = np.zeros(4)
        for k in range(data.ncon):
            c = data.contact[k]
            for i, gid in enumerate(self.foot_geoms):
                if c.geom1 == gid or c.geom2 == gid:
                    mujoco.mj_contactForce(model, data, k, self.f6)
                    fz[i] += self.f6[0]
        return {
            "p": data.qpos[0:3].copy(),
            "quat": data.qpos[3:7].copy(),
            "v_lin": data.qvel[0:3].copy(),       # free joint: linear velocity in world frame
            "omega": data.qvel[3:6].copy(),       # free joint: angular velocity in body frame
            "q": data.qpos[self.qpos_idx].copy(),
            "qd": data.qvel[self.qvel_idx].copy(),
            "fz": fz,
            "req": self.req_name,
            "req_id": self.req_id,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        model, rng = self.model, self.np_random

        # plant mismatch: the controller keeps the nominal constants in self.cfg, only the sim changes
        if self.randomize:
            self.rand = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RAND_RANGES.items()}
        else:
            self.rand = {"mass": 1.0, "friction": float(self.friction_nom[0]), "motor": 1.0, "push": 0.0}
        model.body_mass[self.base_id] = self.base_mass_nom * self.rand["mass"]
        model.geom_friction[self.friction_geoms, 0] = self.rand["friction"] if self.randomize else self.friction_nom

        self._start_pose()
        with silence_output():
            self.core = ControllerCore(self.cfg, top_layer="rl", wbc=self.wbc)
        self.rl = self.core.task.rl
        self.script = list(self.fixed_script) if self.fixed_script is not None else self._random_script()
        self.sent = 0
        self.req_name, self.req_id = None, 0
        self.tau = np.zeros(12)
        self.f6 = np.zeros(6)
        self.walk_t0, self.next_push = None, None
        self.prev_action = np.zeros(N_ACTION)
        self.reward_sums = {k: 0.0 for k in REWARD_SCALES}
        self.stats = {"n": 0, "v_err2": 0.0, "wz_err2": 0.0, "z_err2": 0.0, "z_bias": 0.0, "compute_s": []}

        with silence_output():                  # the WBC's qpOASES builds (and prints) on its first solve
            self._run_until_decision(np.zeros(N_ACTION))
        return self.rl.obs.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.rl.queued_action = action          # taken at the next decision tick (the first tick of this step)
        reward = self._run_until_decision(action)
        self.prev_action = action
        terminated = self.core.fell is not None # type: ignore
        truncated = self.data.time >= self.episode_s
        info = {"rand": dict(self.rand)}
        if terminated or truncated:
            info["episode_reward_terms"] = dict(self.reward_sums)
            info["fell"] = terminated # type: ignore
        return self.rl.obs.copy(), float(reward), terminated, truncated, info

    def _run_until_decision(self, action):
        reward = 0.0
        while True:
            reward += self._tick(action)
            if self.core.fell is not None or self.data.time >= self.episode_s: # type: ignore
                break
            if self.data.time + 1e-9 >= self.rl.next_t:     # the next tick is a decision tick: hand control back
                break
        return max(reward, 0.0)

    def _tick(self, action):
        model, data = self.model, self.data

        while self.script and self.sent < len(self.script) and data.time >= self.script[self.sent][0]:
            self.req_name, self.req_id = self.script[self.sent][1], self.req_id + 1
            self.sent += 1

        t0 = time.perf_counter()
        tau, diag, _ = self.core.step(data.time, self._sensors()) # type: ignore
        self.stats["compute_s"].append(time.perf_counter() - t0)
        data.ctrl[self.act_idx] = self.rand["motor"] * tau

        if self.walk_t0 is None and diag["walk_active"] > 0.5:
            self.walk_t0 = data.time
        if self.rand["push"] > 0.0 and self.walk_t0 is not None:
            if self.next_push is None:
                self.next_push = self.walk_t0 + PUSH_EVERY
            elif data.time >= self.next_push:
                ang = self.np_random.uniform(0.0, 2.0 * np.pi)
                data.qvel[0:2] += self.rand["push"] * self.np_random.uniform(0.5, 1.0) * np.array([np.cos(ang), np.sin(ang)])
                self.next_push += PUSH_EVERY

        # reward on the plant's true state at the start of the tick, held over the tick
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        R = R.reshape(3, 3)
        w_body = data.qvel[3:6]
        w_world = R @ w_body
        wz_world = w_world[2]
        v_cmd = np.array([diag["v_cmd_x"], diag["v_cmd_y"]])
        # MPC state layout: com_pos, rpy, com_vel, omega (world); x y and yaw have weight 0 in the MPC
        com, com_vel = self.rl.model.com_state(data.qpos[0:3], R, data.qpos[self.qpos_idx], data.qvel[0:3],
                                               w_body, data.qvel[self.qvel_idx])
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        x_err = np.concatenate([
            [0.0, 0.0, com[2] - (diag["z_des"] + (com[2] - data.qpos[2]))],   # CoM height ref as in TaskLogic
            [roll, pitch, 0.0],
            com_vel - np.array([v_cmd[0], v_cmd[1], 0.0]),
            w_world - np.array([0.0, 0.0, diag["wz_cmd"]]),
        ])
        terms = {
            "alive": 1.0,
            "mpc_state_cost": float(np.sum(MPC_STATE_Q * x_err ** 2)),
            "mpc_force_cost": MPC_FORCE_R * float(np.sum(self.rl.dist.forces ** 2)),   # forces the top layer sent
        }
        reward = 0.0
        for k, scale in REWARD_SCALES.items():
            r = scale * float(terms[k]) * self.ctrl_dt
            reward += r
            self.reward_sums[k] += r

        # walking quality, same definitions as the SimNode summary (from 1 s after the first walk)
        if self.walk_t0 is not None and self.core.fell is None and data.time >= self.walk_t0 + 1.0: # type: ignore
            dz = data.qpos[2] - diag["z_des"]
            self.stats["n"] += 1
            self.stats["v_err2"] += float(np.sum((data.qvel[0:2] - v_cmd) ** 2))
            self.stats["wz_err2"] += float((wz_world - diag["wz_cmd"]) ** 2)
            self.stats["z_err2"] += float(dz ** 2)
            self.stats["z_bias"] += float(dz)

        for _ in range(CTRL_DECIM):
            mujoco.mj_step(model, data)
        return reward

    def summary(self):
        s = self.stats
        n = max(s["n"], 1)
        compute_ms = 1e3 * np.array(s["compute_s"])
        fell = "no" if self.core.fell is None else f"yes, at t = {self.core.fell:.2f} s" # type: ignore
        print("\n===== ENV SUMMARY =====")
        print(f"fell               : {fell}")
        print(f"speed error RMS    : {np.sqrt(s['v_err2'] / n):.3f} m/s")
        print(f"yaw rate error RMS : {np.sqrt(s['wz_err2'] / n):.3f} rad/s")
        print(f"height error RMS   : {100 * np.sqrt(s['z_err2'] / n):.1f} cm")
        print(f"height bias (mean) : {100 * s['z_bias'] / n:+.2f} cm")
        print(f"controller compute : mean {compute_ms.mean():.2f}   p99 {np.percentile(compute_ms, 99):.2f} ms per 8 ms tick")
        print("reward per term    : " + "   ".join(f"{k} {v:+.2f}" for k, v in self.reward_sums.items()), flush=True)
        self.core.summary() # type: ignore


if __name__ == "__main__":
    # Piece 1 test: zero action on the journal script, no mismatch; must match the WBC + PD baseline
    env = RlTopEnv(randomize=False, script=TEST_SCRIPT, episode_s=25.0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (N_OBS,), obs.shape
    zero = np.zeros(N_ACTION)
    steps, total, done = 0, 0.0, False
    wall0 = time.perf_counter()
    while not done:
        obs, r, terminated, truncated, _ = env.step(zero)
        total += r
        steps += 1
        done = terminated or truncated
    wall = time.perf_counter() - wall0
    print(f"\nenv steps          : {steps} in {wall:.1f} s wall  ->  {steps / wall:.0f} steps/s on one core "
          f"({env.data.time / wall:.1f}x real time)")
    print(f"episode reward     : {total:.2f}")
    print(f"obs finite         : {bool(np.all(np.isfinite(obs)))}")
    env.summary()
