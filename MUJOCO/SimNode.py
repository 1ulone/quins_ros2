import os
import sys
import time
import queue
import threading
import contextlib
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from LOGIC.GaitLogic import LEG_NAMES, JOINT_NAMES
from LOGIC.GaitPlanner import FORWARD
from LOGIC.ControllerCore import ControllerCore, NOMINAL_STANCE
from LINK.Protocol import pack_state, unpack_state, pack_command, unpack_command
from ROS.BaseGUI import GUI

RAND = None             # plant mismatch, same dict format as MujocoSim (RAND_EDGE / RAND_MID)
PUSH_EVERY = 10.0
# SCRIPT = None           # [(time, state), ...] operator requests; None = GUI / gamepad only
SCRIPT = [(2.0, "WALK"), (12.0, "TURN"), (20.0, "IDLE")]
RUN_T = 25.0 
FOOT_NAMES = ('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT')


class LocalLink:
    """Controller inside this process (Piece 1). Same byte packets as the network link, zero delay."""

    def __init__(self, cfg):
        self.core = ControllerCore(cfg)
        self.reply = None

    def send(self, pkt):
        seq, t, t_wall, st = unpack_state(pkt)
        t0 = time.perf_counter()
        tau, diag, state_name = self.core.step(t, st)
        self.reply = pack_command(seq, t_wall, time.perf_counter() - t0, state_name, tau, diag)

    def poll(self):
        # newest command, or None when nothing new arrived
        pkt, self.reply = self.reply, None
        return pkt

    def close(self):
        self.core.summary()


def main(rand=RAND, script=SCRIPT, run_t=RUN_T, headless=False):
    # ---------------- 1. Plant ----------------
    root = Path(__file__).resolve().parent.parent
    model = mujoco.MjModel.from_xml_path(str(root / 'urdf' / 'scene.xml'))
    data = mujoco.MjData(model)

    joints = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    qpos_idx = model.jnt_qposadr[jids]
    qvel_idx = model.jnt_dofadr[jids]
    act_idx = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor") for j in joints])
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f) for f in FOOT_NAMES]
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_collision") for f in FOOT_NAMES]
    lf_hip_x = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'LF_HIP')][0]
    assert np.sign(lf_hip_x) == FORWARD, f"LF_HIP is at x = {lf_hip_x:+.3f} but FORWARD = {FORWARD}"
    joint_damping = model.dof_damping[qvel_idx]
    assert np.allclose(joint_damping, joint_damping[0]), "WBC expects one damping value for all joints"

    # Start in NOMINAL_STANCE with the feet just clear of the floor
    data.qpos[qpos_idx] = NOMINAL_STANCE
    mujoco.mj_forward(model, data)
    data.qpos[2] -= min(data.xpos[f][2] for f in foot_ids)
    mujoco.mj_forward(model, data)
    while data.ncon > 0:
        data.qpos[2] += 0.01
        mujoco.mj_forward(model, data)
    foot_r = model.geom_size[foot_geoms[0]][0]
    z_nom = float(data.qpos[2]) - (min(data.xpos[f][2] for f in foot_ids) - foot_r)
    foot_nom_xy = np.array([data.xpos[f][0:2] - data.qpos[0:2] for f in foot_ids])

    # Robot constants the controller needs (nominal model, before any mismatch below)
    cfg = {
        "ctrl_dt": float(model.opt.timestep),
        "z_nom": z_nom,
        "foot_nom_xy": foot_nom_xy,
        "foot_r": float(foot_r),
        "robot_mass": float(model.body_mass.sum()),
        "joint_damping": float(joint_damping[0]),
    }

    print(f"[episode] conditions = {rand if rand else 'nominal'}", flush=True)
    rng = np.random.default_rng(rand["seed"]) if rand else None
    motor, next_push = 1.0, None
    if rand:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        model.body_mass[base_id] *= rand["mass"]
        floor_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('floor', 'uneven_floor')]
        for gid in [g for g in floor_geoms if g >= 0] + foot_geoms:
            model.geom_friction[gid][0] = rand["friction"]
        motor = rand["motor"]

    # ---------------- 2. Input (operator side) ----------------
    req = {"name": None, "id": 0}

    def request_state(name):
        # GUI / gamepad / script only store the newest request; it rides on every state packet
        req["name"] = name
        req["id"] += 1

    graph_queue = queue.Queue()

    def start_gui():
        gui = GUI({"state": request_state})
        gui.setup()
        while True:
            redraw = False
            while not graph_queue.empty():
                g = graph_queue.get_nowait()
                gui.time_history.append(g[0])
                gui.desired_history.append(g[1])
                gui.measured_history.append(g[2])
                redraw = True
            if redraw:
                gui.refresh_graph()
            gui.update()
            time.sleep(0.05)

    if not headless:
        threading.Thread(target=start_gui, daemon=True).start()

    # ---------------- 3. Link to the controller ----------------
    link = LocalLink(cfg)

    physics_hz = 1.0 / model.opt.timestep
    print_dec = int(physics_hz / 2)
    graph_dec = int(physics_hz / 50)
    tau = np.zeros(len(joints))
    diag = None
    ctrl_state = "TUNING"
    seq, applied_seq, sent = 0, -1, 0
    f6 = np.zeros(6)
    R_flat = np.zeros(9)
    walk_t0, fell = None, None
    stats = {"n": 0, "v_err2": 0.0, "wz_err2": 0.0, "z_err2": 0.0, "z_bias": 0.0}
    step = 0
    last_render = time.perf_counter()

    try:
        with (contextlib.nullcontext() if headless else mujoco.viewer.launch_passive(model, data)) as viewer:
            if viewer is not None:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

            while headless or viewer.is_running():  # type: ignore
                step_start = time.perf_counter()

                # ---- Input: scripted requests ----
                while script and sent < len(script) and data.time >= script[sent][0]:
                    request_state(script[sent][1])
                    sent += 1

                # ---- Sensors -> state packet ----
                fz = np.zeros(4)
                for k in range(data.ncon):
                    c = data.contact[k]
                    for i, gid in enumerate(foot_geoms):
                        if c.geom1 == gid or c.geom2 == gid:
                            mujoco.mj_contactForce(model, data, k, f6)
                            fz[i] += f6[0]
                req_id = req["id"]
                st = {
                    "p": data.qpos[0:3],
                    "quat": data.qpos[3:7],
                    "v_lin": data.qvel[0:3],        # free joint: linear velocity in world frame
                    "omega": data.qvel[3:6],        # free joint: angular velocity in body frame
                    "q": data.qpos[qpos_idx],
                    "qd": data.qvel[qvel_idx],
                    "fz": fz,
                    "req": req["name"],
                    "req_id": req_id,
                }
                link.send(pack_state(seq, data.time, time.perf_counter(), st))
                seq += 1

                # ---- Newest command; never wait, hold the last torques until a newer one arrives ----
                pkt = link.poll()
                if pkt is not None:
                    answered_seq, _, _, state_name, cmd_tau, cmd_diag = unpack_command(pkt)
                    if answered_seq > applied_seq:
                        applied_seq, tau, diag, ctrl_state = answered_seq, cmd_tau, cmd_diag, state_name
                data.ctrl[act_idx] = motor * tau

                if diag is not None:
                    if walk_t0 is None and diag["walk_active"] > 0.5:
                        walk_t0 = data.time
                    if fell is None and diag["fell_t"] >= 0.0:
                        fell = diag["fell_t"]

                    # ---- Walking quality, from 1 s after the first walk (skips the stride ramp) ----
                    if walk_t0 is not None and fell is None and data.time >= walk_t0 + 1.0:
                        mujoco.mju_quat2Mat(R_flat, data.qpos[3:7])
                        wz_world = (R_flat.reshape(3, 3) @ data.qvel[3:6])[2]
                        dz = data.qpos[2] - diag["z_des"]
                        stats["n"] += 1
                        stats["v_err2"] += float((data.qvel[0] - diag["v_cmd_x"]) ** 2 + (data.qvel[1] - diag["v_cmd_y"]) ** 2)
                        stats["wz_err2"] += float((wz_world - diag["wz_cmd"]) ** 2)
                        stats["z_err2"] += float(dz ** 2)
                        stats["z_bias"] += float(dz)

                    # ---- Pushes (plant disturbance) ----
                    if rand and rand["push"] > 0.0 and walk_t0 is not None:
                        if next_push is None:
                            next_push = walk_t0 + PUSH_EVERY
                        elif data.time >= next_push:
                            ang = rng.uniform(0.0, 2.0 * np.pi)  # type: ignore
                            data.qvel[0:2] += rand["push"] * rng.uniform(0.5, 1.0) * np.array([np.cos(ang), np.sin(ang)])  # type: ignore
                            next_push += PUSH_EVERY

                    if not headless and step % graph_dec == 0:
                        graph_queue.put([float(data.time), diag["fl_z_des"], float(data.xpos[foot_ids[0]][2])])

                    if not headless and step % print_dec == 0:
                        st_str = ''.join('S' if diag[f"st_{leg}"] > 0.5 else '-' for leg in LEG_NAMES)
                        print(f"t={data.time:.2f} {ctrl_state:<10} ok={diag['wbc_ok'] > 0.5} st={st_str} "
                              f"v=({data.qvel[0]:+.3f},{data.qvel[1]:+.3f}) cmd=({diag['v_cmd_x']:+.3f},{diag['v_cmd_y']:+.3f}) "
                              f"z={data.qpos[2]:.3f} yaw={diag['yaw']:+.3f}/{diag['target_yaw']:+.3f} "
                              f"|s|={diag['s_max']:.3f} frz={int(diag['frozen'])}", flush=True)

                mujoco.mj_step(model, data)

                if not headless and time.perf_counter() - last_render > 0.016:
                    viewer.sync()  # type: ignore
                    last_render = time.perf_counter()

                step += 1
                if (headless and fell is not None and data.time > fell + 1.0) or (run_t is not None and data.time >= run_t):
                    break
                elapsed = time.perf_counter() - step_start
                if not headless and elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    finally:
        link.close()
        n = max(stats["n"], 1)
        print("\n===== SUMMARY =====", flush=True)
        print(f"fell               : {'no' if fell is None else f'yes, at t = {fell:.2f} s'}")
        print(f"speed error RMS    : {np.sqrt(stats['v_err2'] / n):.3f} m/s")
        print(f"yaw rate error RMS : {np.sqrt(stats['wz_err2'] / n):.3f} rad/s")
        print(f"height error RMS   : {100 * np.sqrt(stats['z_err2'] / n):.1f} cm")
        print(f"height bias (mean) : {100 * stats['z_bias'] / n:+.2f} cm", flush=True)


if __name__ == '__main__':
    main()
