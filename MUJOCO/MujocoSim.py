import os
import sys
import time
import queue
import threading
import math as m
import pandas as pd
from pathlib import Path

import imageio
import mujoco
import mujoco.viewer
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.WbcLogic import WholeBodyController
from LOGIC.TaskLogic import TaskController
from LOGIC.GaitPlanner import FORWARD
from ROS.BaseGUI import GUI

NOMINAL_STANCE = np.tile([0.0, -0.83, 1.113], 4)
WBC_DEC = 1     # run the WBC every N physics steps (1 = every step)
USE_FOSMC = True 
FOSMC_TEST_A = False 
AUTO_WALK_T = 2.0
AUTO_STATE = None 
RUN_T = None 
GROUP_ROWS = {"lin": slice(0, 3), "ang": slice(3, 6), "swing": slice(6, 18)}
WALK_DROP = 0.085
IDLE_DROP = 0.05

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
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_collision") for f in ('LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT')]
    # The front legs must sit on the FORWARD side of the body, or the gait walks against itself
    lf_hip_x = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'LF_HIP')][0]
    assert np.sign(lf_hip_x) == FORWARD, f"LF_HIP is at x = {lf_hip_x:+.3f} but FORWARD = {FORWARD}"

    # The WBC's RNEA must use the same joint damping as the simulator (eq. 5)
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

    # ---------------- 2. Logic & GUI ----------------
    graph_queue = queue.Queue()
    logic = GaitLogic()
    logic.robot_mass = float(model.body_mass.sum())

    def start_gui():
        gui = GUI({
            "state": logic.request_state,
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
    wbc = WholeBodyController(urdf_path, joints_name_list, joint_damping=float(joint_damping[0]))
    task = TaskController(
        logic,
        NOMINAL_STANCE,
        z_nom,
        foot_nom_xy,
        use_fosmc=USE_FOSMC,
        fosmc_dt=model.opt.timestep * WBC_DEC,
        walk_drop=WALK_DROP,
        idle_drop=IDLE_DROP,
    )
    print(f"[init] walk pose (FL) = {np.round(task.q_walk[0:3], 3)}", flush=True)

    if task.fosmc is not None and FOSMC_TEST_A:
        task.fosmc.update_gains(lam=0.95)

    physics_hz = 1.0 / model.opt.timestep
    logic_dec = int(physics_hz / logic.control_rate)
    print_dec = int(physics_hz / 2)

    record_hz = 60
    record_dec = int(physics_hz / record_hz)
    renderer = mujoco.Renderer(model, height=480, width=640)
    video_writer = imageio.get_writer('simulation.mp4', fps=record_hz)

    tau = np.zeros(len(joints_name_list))
    v_base = np.zeros(6)
    continuous_yaw, prev_raw_yaw = 0.0, 0.0
    step = 0
    last_render = time.time()
    # Diagnostics, collected from 1 s after WALK starts (skips the stride ramp)
    stats = {"n": 0, "n_st": 0, "n_sw": 0, "v_err2": 0.0, "wz_err2": 0.0, "z_err2": 0.0, "z_bias": 0.0, "late": 0, "early": 0,
             "fell": None, "s": {g: 0.0 for g in GROUP_ROWS}, "miss": {g: 0.0 for g in GROUP_ROWS},
             "cnt": {g: 0 for g in GROUP_ROWS}}
    win = {g: [0.0, 0.0] for g in GROUP_ROWS}   # per print window: peak |s|, peak mean miss
    walk_t0 = None
    log = []

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

                # ---- Gait logic (schedule + heading) ----
                if step % logic_dec == 0:
                    raw_yaw = m.atan2(R_act[1, 0], R_act[0, 0])
                    continuous_yaw += (raw_yaw - prev_raw_yaw + m.pi) % (2.0 * m.pi) - m.pi
                    prev_raw_yaw = raw_yaw
                    logic.current_q, logic.current_q_dot = q_act, qd_act
                    logic.current_yaw = continuous_yaw
                    if AUTO_STATE is not None and AUTO_WALK_T is not None and logic.current_state == "TUNING" and data.time >= AUTO_WALK_T:
                        logic.update_state(AUTO_STATE)
                        walk_t0 = data.time
                    logic.loop_step(data.time)

                # ---- WBC (synchronous, sim time only) ----
                if step % WBC_DEC == 0:
                    p = data.qpos[0:3].copy()
                    v_base = np.concatenate([data.qvel[0:3], R_act @ data.qvel[3:6]])   # free joint omega is body frame
                    wbc.update_state(p=p, R=R_act, q=q_act, v_base=v_base, v_joints=qd_act)
                    cmd = task.compute(t=data.time, dt=model.opt.timestep * WBC_DEC,
                                       p=p, R=R_act, v_base=v_base, q=q_act, v_joints=qd_act,
                                       pf=wbc.pf, vf=wbc.vf,
                                       acc_achieved=np.concatenate([wbc.last["acc_base"], wbc.last["acc_foot"]]))
                    tau = wbc.solve(**cmd)["tau"]
                    # ---- Diagnostics ----
                    g1, g2 = data.contact.geom1[:data.ncon], data.contact.geom2[:data.ncon]
                    in_contact = np.array([np.any(g1 == gid) or np.any(g2 == gid) for gid in foot_geoms])
                    dbg = task.debug
                    for g, rows in GROUP_ROWS.items():
                        act = dbg["active"][rows]
                        if np.any(act):
                            win[g][0] = max(win[g][0], float(np.max(np.abs(dbg["s"][rows][act]))))
                            win[g][1] = max(win[g][1], float(np.mean(dbg["miss"][rows][act])))
                    if walk_t0 is not None and stats["fell"] is None and data.time >= walk_t0 + 1.0:
                        sched = dbg["stance"] > 0.5
                        stats["n"] += 1
                        stats["n_st"] += int(np.sum(sched))
                        stats["n_sw"] += int(np.sum(~sched))
                        stats["v_err2"] += float(np.sum((data.qvel[0:2] - task.v_cmd) ** 2))
                        w_cmd = logic.yaw_rate if logic.turning else 0.0
                        stats["wz_err2"] += float((v_base[5] - w_cmd) ** 2)
                        z_tgt = task.z_target()
                        stats["z_err2"] += float((data.qpos[2] - z_tgt) ** 2)
                        stats["z_bias"] += float(data.qpos[2] - z_tgt)
                        stats["late"] += int(np.sum(sched & ~in_contact))
                        stats["early"] += int(np.sum(~sched & in_contact))
                        for g, rows in GROUP_ROWS.items():
                            act = dbg["active"][rows]
                            if np.any(act):
                                stats["s"][g] += float(np.mean(np.abs(dbg["s"][rows][act])))
                                stats["miss"][g] += float(np.mean(dbg["miss"][rows][act]))
                                stats["cnt"][g] += 1
                    if stats["fell"] is None and (data.qpos[2] < 0.2 or R_act[2, 2] < 0.5):
                        stats["fell"] = data.time

                data.ctrl[act_idx] = tau

                # Graph: FL foot height, target vs actual
                if step % logic_dec == 0:
                    graph_queue.put([float(data.time), float(task.debug["pf_d"][2]), float(wbc.pf[2])])
                    # ---- Data log (Excel export): target vs actual per task, |s| per gain group ----
                    dbg = task.debug
                    s_abs = np.abs(dbg["s"])
                    row = {"t": data.time, "state": logic.current_state,
                           "stance": ''.join('S' if s > 0.5 else '-' for s in dbg["stance"]),
                           "FL_z_des": dbg["pf_d"][2], "FL_z": wbc.pf[2],
                           "z_des": task.z_target(), "z": data.qpos[2],
                           "vx_cmd": task.v_cmd[0], "vx": data.qvel[0],
                           "vy_cmd": task.v_cmd[1], "vy": data.qvel[1],
                           "yaw_des": task.yaw_d, "yaw": continuous_yaw,
                           "wz_cmd": logic.yaw_rate if logic.turning else 0.0, "wz": v_base[5]}
                    for g, rows in GROUP_ROWS.items():
                        act = dbg["active"][rows]
                        row[f"s_{g}"] = float(np.mean(s_abs[rows][act])) if np.any(act) else 0.0
                    log.append(row)

                if step % print_dec == 0:
                    st = ''.join('S' if s > 0.5 else '-' for s in task.debug["stance"])
                    fz = wbc.last["Fc"][2::3].sum()
                    print(f"t={data.time:.2f} {logic.current_state:<6} ok={wbc.ok} st={st} "
                          f"v=({data.qvel[0]:+.3f},{data.qvel[1]:+.3f}) cmd=({task.v_cmd[0]:+.3f},{task.v_cmd[1]:+.3f}) "
                          f"z={data.qpos[2]:.3f} yaw={continuous_yaw:+.3f}/{logic.target_yaw:+.3f} Fz={fz:.1f} "
                          f"|s|={task.debug['s_max']:.3f} frz={task.debug['frozen']}", flush=True)
                    print("      peak |s| lin/ang/swing = " + "/".join(f"{win[g][0]:.2f}" for g in GROUP_ROWS)
                          + "   peak miss lin/ang/swing = " + "/".join(f"{win[g][1]:.2f}" for g in GROUP_ROWS), flush=True)
                    win = {g: [0.0, 0.0] for g in GROUP_ROWS}

                mujoco.mj_step(model, data)

                if time.time() - last_render > 0.016:
                    viewer.sync()
                    last_render = time.time()
                if step % record_dec == 0:
                    mujoco.mjv_updateScene(model, data, viewer.opt, None, viewer.cam,
                                           mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
                    video_writer.append_data(renderer.render())

                step += 1
                if (stats["fell"] is not None and data.time > stats["fell"] + 1.0) or (RUN_T is not None and data.time >= RUN_T):
                    break
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
    finally:
        video_writer.close()
        renderer.close()
        if log:
            pd.DataFrame(log).to_excel("sim_log.xlsx", index=False)
            print(f"[log] {len(log)} rows saved to sim_log.xlsx", flush=True)
        n = max(stats["n"], 1)
        fell = "no" if stats["fell"] is None else f"yes, at t = {stats['fell']:.2f} s"
        print("\n===== SUMMARY =====", flush=True)
        print(f"fell               : {fell}")
        print(f"speed error RMS    : {np.sqrt(stats['v_err2'] / n):.3f} m/s")
        print(f"yaw rate error RMS : {np.sqrt(stats['wz_err2'] / n):.3f} rad/s")
        print(f"height error RMS   : {100 * np.sqrt(stats['z_err2'] / n):.1f} cm")
        print(f"height bias (mean) : {100 * stats['z_bias'] / n:+.2f} cm")
        print(f"stance, no contact : {100 * stats['late'] / max(stats['n_st'], 1):.1f} %")
        print(f"swing, in contact  : {100 * stats['early'] / max(stats['n_sw'], 1):.1f} %")
        for g in GROUP_ROWS:
            c = max(stats["cnt"][g], 1)
            print(f"{g:<6} mean |s| = {stats['s'][g] / c:.3f}   mean miss = {stats['miss'][g] / c:.2f}", flush=True)


if __name__ == '__main__':
    main()
