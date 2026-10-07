import os
import sys
import time
import queue
import threading
import math as m
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from LOGIC.GaitPlanner import GaitPlanner
from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.FOSMCLogic import FOSMC
from LOGIC.MpcLogic import WholeBodyMPC
from ROS.BaseGUI import GUI

NOMINAL_STANCE = np.tile([0.0, 0.60, -1.10], 4) 
USE_FOSMC = False
MPC_HZ = 50
MPC_WEIGHTS = [100.0, 1700.0, 0.04, 4.4e-7, 0.0004]   # qj, v_base, v_j, tau, a
BASE_W = (47000.0, 3000.0)                            # base height, base orientation
SWING_W = 5000.0


def yaw_rot(psi):
    c, s = m.cos(psi), m.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main():
    # ---------------- 1. Model ----------------
    script_dir = Path(__file__).resolve().parent
    scene_path = str(script_dir.parent / 'urdf' / 'scene.xml')
    urdf_path = str(script_dir.parent / 'urdf' / 'quadruped.urdf')

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    joints_name_list = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints_name_list]
    qpos_idx = model.jnt_qposadr[jids]
    qvel_idx = model.jnt_dofadr[jids]
    act_idx = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor") for j in joints_name_list])
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f) for f in ('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT')]

    # The MPC's RNEA must use the same joint damping as the simulator (eq. 5)
    joint_damping = model.dof_damping[qvel_idx]
    assert np.allclose(joint_damping, joint_damping[0]), "MPC expects one damping value for all joints"

    # Start in NOMINAL_STANCE with the feet just clear of the floor
    data.qpos[qpos_idx] = NOMINAL_STANCE
    mujoco.mj_forward(model, data)
    data.qpos[2] -= min(data.xpos[f][2] for f in foot_ids)
    mujoco.mj_forward(model, data)
    while data.ncon > 0:
        data.qpos[2] += 0.01
        mujoco.mj_forward(model, data)
    z_nom = float(data.qpos[2])
    foot_nom_xy = np.array([data.xpos[f][0:2] - data.qpos[0:2] for f in foot_ids])
    print(f"[init] base z = {z_nom:.3f}", flush=True)

    # ---------------- 2. Logic & GUI ----------------
    graph_queue = queue.Queue()
    logic = GaitLogic()
    logic.robot_mass = float(model.body_mass.sum())

    def start_gui():
        gui = GUI({
            "state": logic.update_state,
            "phase": logic.update_phase_offsets,
            "wt_params": logic.update_wt_params,
            "jt_params": logic.update_jt_params,
            "raw_tune": logic.raw_tune,
            "gamepad": logic.update_gamepad_params,
        })
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

    threading.Thread(target=start_gui, daemon=True).start()

    # ---------------- 3. Controllers ----------------
    mpc = WholeBodyMPC(
        urdf_path,
        joints_name_list,
        n=37,
        dt=0.015,
        joint_damping=float(joint_damping[0]),
        base_w=BASE_W,
        swing_w=SWING_W
    )

    fosmc = FOSMC(dof=12, dt=model.opt.timestep, lam=0.5, alpha=1.5, Ke1=0.5, Ke2=0.5,
                  Ks=5.0, Kr=2.0, gamma_c=0.01, gamma_a=0.01) if USE_FOSMC else None

    physics_hz = 1.0 / model.opt.timestep
    logic_dec = int(physics_hz / logic.control_rate)
    mpc_dec = int(physics_hz / MPC_HZ)
    print_dec = int(physics_hz / 2)

    record_hz = 60
    record_dec = int(physics_hz / record_hz)
    renderer = mujoco.Renderer(model, height=480, width=640)
    video_writer = imageio.get_writer('simulation.mp4', fps=record_hz)

    # Trajectory currently being executed (hold pose until the first solve)
    traj = {
        "tau": np.zeros((mpc.nj, mpc.n)),
        "qj": np.tile(data.qpos[qpos_idx].reshape(-1, 1), (1, mpc.n + 1)),
        "vj": np.zeros((mpc.nj, mpc.n + 1)),
        "t0": data.time,
    }

    turn_anchor_xy = np.zeros(2)
    was_turning = False
    continuous_yaw, prev_raw_yaw = 0.0, 0.0
    target_v = np.zeros(2)
    step = 0
    last_render = time.time()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

            while viewer.is_running():
                step_start = time.time()
                q_act = data.qpos[qpos_idx].copy()
                qd_act = data.qvel[qvel_idx].copy()

                R_act = np.zeros(9)
                mujoco.mju_quat2Mat(R_act, data.qpos[3:7])
                R_act = R_act.reshape(3, 3)

                if logic.turning and not was_turning:
                    turn_anchor_xy = data.qpos[0:2].copy()
                was_turning = logic.turning

                # ---- Gait logic (schedule + heading) ----
                if step % logic_dec == 0:
                    raw_yaw = m.atan2(R_act[1, 0], R_act[0, 0])
                    continuous_yaw += (raw_yaw - prev_raw_yaw + m.pi) % (2.0 * m.pi) - m.pi
                    prev_raw_yaw = raw_yaw
                    logic.current_q, logic.current_q_dot = q_act, qd_act
                    logic.current_yaw = continuous_yaw
                    logic.loop_step(data.time)

                # ---- MPC (synchronous: the sim waits for the solve) ----
                if step % mpc_dec == 0:
                    target_v[:] = 0.0
                    if logic.walking and isinstance(logic.planner, GaitPlanner):
                        gp = logic.planner.p
                        v_fwd = -(gp.step_len * logic.planner.stride_scale * gp.freq) / gp.duty   # body -x is forward
                        target_v[:] = v_fwd * np.array([m.cos(logic.target_yaw), m.sin(logic.target_yaw)])
                    elif logic.turning:
                        target_v[:] = np.clip(1.5 * (turn_anchor_xy - data.qpos[0:2]), -0.2, 0.2)

                    horizon = logic.mpc_horizon(data.time, mpc.n, mpc.dt)
                    # swing x, y targets: base moving at the commanded velocity + planner offset from the nominal foot
                    R_yaw = yaw_rot(logic.target_yaw)[0:2, 0:2]
                    swing_xy = np.zeros((8, mpc.n))
                    for k in range(mpc.n):
                        base_xy = data.qpos[0:2] + target_v * k * mpc.dt
                        for i in range(4):
                            swing_xy[2 * i: 2 * i + 2, k] = base_xy + R_yaw @ (foot_nom_xy[i] + np.array(horizon["swing_xy"][k][i]))

                    v0 = np.concatenate([data.qvel[0:3], R_act @ data.qvel[3:6], qd_act])   # MuJoCo free-joint omega is body frame

                    try:
                        tau, qj, vj = mpc.solve(
                            p0=data.qpos[0:3].copy(), R0=R_act, qj0=q_act, v0=v0,
                            q_nom=NOMINAL_STANCE,
                            v_base_des=np.array([target_v[0], target_v[1], 0.0, 0.0, 0.0, logic.yaw_rate]),
                            swing_vz_schedule=np.array(horizon["swing_vz"]).T,
                            stance_schedule=np.array(horizon["stance"]).T,
                            mpc_weights=MPC_WEIGHTS,
                            vz_active_schedule=np.array(horizon["vz_active"]).T,
                            fc_guess=np.array(horizon["fc_guess"]).reshape(mpc.n, 12).T,
                            z_des=z_nom,
                            R_des=yaw_rot(logic.target_yaw),
                            swing_xy=swing_xy,
                            xy_active=np.array(horizon["xy_active"]).T,
                        )
                        traj.update(tau=tau, qj=qj, vj=vj, t0=data.time)
                    except RuntimeError as e:
                        print(f"[mpc t={data.time:.2f}] solve failed, keeping previous plan: {e}", flush=True)

                # ---- Interpolate the plan (eq. 9) ----
                k_f = np.clip((data.time - traj["t0"]) / mpc.dt, 0.0, mpc.n)
                k0 = int(k_f)
                k1 = min(k0 + 1, mpc.n)
                a = k_f - k0
                q_des = (1 - a) * traj["qj"][:, k0] + a * traj["qj"][:, k1]
                qd_des = (1 - a) * traj["vj"][:, k0] + a * traj["vj"][:, k1]
                tau = traj["tau"][:, min(k0, mpc.n - 1)].copy()

                if fosmc is not None:
                    tau += fosmc.compute(q=q_act, q_dot=qd_act, q_d=q_des, q_dot_d=qd_des)

                data.ctrl[act_idx] = tau

                if step % logic_dec == 0:
                    graph_queue.put([float(data.time), float(q_des[1]), float(q_act[1])])

                if step % print_dec == 0:
                    print(f"t={data.time:.2f} {logic.current_state:<6} conv={mpc.converged} "
                          f"v=({data.qvel[0]:+.3f},{data.qvel[1]:+.3f}) cmd=({target_v[0]:+.3f},{target_v[1]:+.3f}) "
                          f"z={data.qpos[2]:.3f} yaw={continuous_yaw:+.3f}/{logic.target_yaw:+.3f}", flush=True)

                mujoco.mj_step(model, data)

                if time.time() - last_render > 0.016:
                    viewer.sync()
                    last_render = time.time()
                if step % record_dec == 0:
                    mujoco.mjv_updateScene(model, data, viewer.opt, None, viewer.cam,
                                           mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
                    video_writer.append_data(renderer.render())

                step += 1
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    finally:
        video_writer.close()


if __name__ == '__main__':
    main()
