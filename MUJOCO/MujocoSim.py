import os
import sys
import time
import queue
import mujoco
import threading
import imageio
import mujoco.viewer
import math as m
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from LOGIC.GaitPlanner import GaitPlanner
from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.FOSMCLogic import FOSMC
from LOGIC.MpcLogic import WholeBodyMPC 
from ROS.BaseGUI import GUI
from pathlib import Path

NOMINAL_STANCE = np.tile([0.0, 0.60, -1.10], 4)
SYNC_MPC = True
USE_FOSMC = False

def main():
    # ---------------- 1. Setup Mujoco Environment ----------------
    script_dir = Path(__file__).resolve().parent
    scene_path = str(script_dir.parent/'urdf'/'scene.xml')
    urdf_path = str(script_dir.parent/'urdf'/'quadruped.urdf')
    
    if not os.path.exists(scene_path):
        print(f"Error: {scene_path} not found. Run convert_urdf.py first.")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    joint_info = {}
    for leg, joints in JOINT_NAMES.items():
        for joint in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            joint_info[joint] = {
                'qpos_adr': model.jnt_qposadr[jid],
                'qvel_adr': model.jnt_dofadr[jid],
                'actuator_id': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_motor"),
            }

    foot_body_names = ['tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link']
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names]
    hip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ['tl_thigh_link', 'tr_thigh_link', 'bl_thigh_link', 'br_thigh_link']]
    scuff_steps = np.zeros(4, dtype=int)

    # IDLE test init: start in NOMINAL_STANCE, feet just clear of the floor.
    # Raise the base in small steps until nothing touches, so no floor height or foot radius is assumed.
    START_AT_NOMINAL = True
    if START_AT_NOMINAL:
        idx = 0
        for leg in LEG_NAMES:
            for j in JOINT_NAMES[leg]:
                data.qpos[joint_info[j]['qpos_adr']] = NOMINAL_STANCE[idx]
                idx += 1
        mujoco.mj_forward(model, data)
        lowest_foot = min(data.xpos[fid][2] for fid in foot_ids)
        data.qpos[2] -= lowest_foot            # coarse: foot origins roughly at z = 0
        mujoco.mj_forward(model, data)
        for _ in range(1000):                   # fine: lift 0.01 per step until no contact
            if data.ncon == 0:
                break
            data.qpos[2] += 0.01
            mujoco.mj_forward(model, data)
        print(f"[init] base z = {data.qpos[2]:.3f}, contacts = {data.ncon}", flush=True)

    z_nom = float(data.qpos[2])
    # sensor_names = ['tl_foot_force_sensor', 'tr_foot_force_sensor', 'bl_foot_force_sensor', 'br_foot_force_sensor']
    # sensor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in sensor_names]
    # ---------------- 2. Setup Logic & UI ----------------
    
    cmd = {
        "q_des": np.tile([0.0, 0.45, -0.60], 4),
        "qd_des": np.zeros(12),
        "foot_forces": np.zeros((4, 3)),
        "is_stance": [False, False, False, False],
    }

    graph_queue = queue.Queue()

    def handle_walk_points(points_data):
        if not points_data: return
        pt = points_data[0] 
        cmd["q_des"] = np.array(pt["positions"])
        cmd["qd_des"] = np.array(pt["velocities"])
        cmd["is_stance"] = pt["is_stance"]
        cmd["foot_forces"] = pt["foot_forces"]

    def handle_jump_points(q_desired, qd_desired, foot_forces, is_stance):
        cmd["q_des"] = np.array(q_desired)
        cmd["qd_des"] = np.array(qd_desired)
        cmd["foot_forces"] = np.array(foot_forces)
        cmd["is_stance"] = is_stance

    def handle_transition(current_angles):
        cmd["q_des"] = np.array(current_angles)
        cmd["qd_des"] = np.zeros(12)
        
    def handle_raw_tune(raw_angles):
        coxa, femur, tibia = raw_angles
        positions = []
        for _ in range(4):
            positions.extend([coxa, femur, tibia])
        cmd["q_des"] = np.array(positions)
        cmd["qd_des"] = np.zeros(12)
        
    def handle_graph_push(graph_data):
        graph_queue.put(graph_data)

    logic = GaitLogic({
        "walk_points": handle_walk_points,
        "jump_points": handle_jump_points,
        "transition_cb": handle_transition,
        "raw_tune_cb": handle_raw_tune,
        "graph": handle_graph_push
    })

    shared_mpc_weights = [100.0, 25.0, 0.04, 4.4e-7, 0.0004]

    # def handle_mpc_params(weights):
    #     for i in range(5):
    #         shared_mpc_weights[i] = weights[i]

    def start_gui():
        gui = GUI({
            "state": logic.update_state,
            "phase": logic.update_phase_offsets,
            "wt_params": logic.update_wt_params,
            "jt_params": logic.update_jt_params,
            "raw_tune": logic.raw_tune,
            "gamepad": logic.update_gamepad_params,
            # "fosmc_params": lambda gains: fosmc.update_gains(*gains),
            # "mpc_params": handle_mpc_params,
        })
        gui.setup()
        
        while True:
            needs_redraw = False
            
            while not graph_queue.empty():
                g_data = graph_queue.get_nowait()
                gui.time_history.append(g_data[0])
                gui.desired_history.append(g_data[1])
                gui.measured_history.append(g_data[2])
                needs_redraw = True
                
            if needs_redraw:
                gui.refresh_graph()
                
            gui.update()
            time.sleep(0.05)

    threading.Thread(target=start_gui, daemon=True).start()

    # ---------------- 3. Simulation Loop ----------------
    physics_hz = 1.0 / model.opt.timestep 
    control_hz = logic.control_rate       
    decimation_steps = int(physics_hz / control_hz)
    step_counter = 0

    fosmc = FOSMC(
        dof=12,
        dt=model.opt.timestep,
        lam=0.5,
        alpha=1.5,
        Ke1=0.5,
        Ke2=0.5,
        Ks=5.0,
        Kr=2.0,
        gamma_c=0.01,
        gamma_a=0.01
    )

    record_hz = 60
    record_steps = int(physics_hz / record_hz)
    renderer = mujoco.Renderer(model, height=480, width=640)
    video_writer = imageio.get_writer('simulation.mp4', fps=record_hz)

    last_render_time = time.time()

    joints_name_list = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
    mpc = WholeBodyMPC(urdf_path, joints_name_list, n=14, dt=0.04)

    physics_hz = int(1.0 / model.opt.timestep)
    mpc_hz = 50
    mpc_decimation = int(physics_hz / mpc_hz)


    # ---------------- 4. Asynchronous MPC Setup ----------------
    mpc_lock = threading.Lock()
    
    # Shared state dictionary for the MPC thread to read from
    shared_mpc_state = {
        "p_act": np.zeros(3),
        "R_act": np.eye(3),
        "v_act": np.zeros(3),
        "rpy_act": np.zeros(3),
        "w_act": np.zeros(3),
        "qj_act": np.zeros(12),
        "vj_act": np.zeros(12),
        "q_des": np.zeros(12),
        "qd_des": np.zeros(12),
        "foot_positions": [np.zeros(3) for _ in range(4)],
        "horizon": logic.mpc_horizon(data.time, mpc.n, mpc.dt),
        "z_off": 2.5,
        "target_vx": 0.0,
        "target_vy": 0.0,
        "target_yaw": 0.0,
        "yaw_rate": 0.0,
        "p_ref_xy": np.zeros(2),
        "t_snap": data.time
    }
    
    qj0_init = np.zeros(12)
    idx = 0
    for leg in LEG_NAMES:
        for j in JOINT_NAMES[leg]:
            qj0_init[idx] = data.qpos[joint_info[j]['qpos_adr']]
            idx += 1

    # Placeholder for before the first solve completes. Shape has to match what
    # mpc_worker now publishes: a full (n) torque horizon and (n+1) position/
    # velocity horizons, not a single vector -- otherwise the very first physics
    # tick crashes trying to index a trajectory that isn't there yet.
    mpc_output = {
        "tau_traj": np.zeros((mpc.nj, mpc.n)),                      # (12, 3): no torque yet
        "qj_traj": np.tile(qj0_init.reshape(-1, 1), (1, mpc.n + 1)), # (12, 4): hold current pose
        "v_traj": np.zeros((mpc.nj, mpc.n + 1)),                    # (12, 4): zero velocity
        "t0": data.time,
        "failed_t": -1.0,
    }

    turn_anchor_xy = np.zeros(2)
    was_turning = False
    mpc_running = True

    # DEBUG: proves whether solves are happening, how often, and whether they raise
    mpc_debug = {"solve_count": 0, "last_solve_t": time.time(), "last_exc": None}

    def mpc_worker():
        last_solve_t = -1.0
        def solve_summary(t_snap, stance_arr, tag):
            try:
                st_ = mpc.opti.stats()
            except Exception:
                st_ = {}
            legs = ("FL", "FR", "BL", "BR")
            sched = np.array(stance_arr, dtype=bool)
            pattern = " ".join(legs[i] + ":" + "".join("S" if s else "w" for s in sched[i]) for i in range(4))
            print(
                f"[mpc t={t_snap:.2f} {tag} iters={st_.get('iter_count')} status={st_.get('return_status')}]"
                f"wall={st_.get('t_wall_total', float('nan')):.3f}s N={st_.get('N')} | {pattern}"
            , flush=True)
        while mpc_running:
            start_t = time.time()
            
            # Safely copy current state
            with mpc_lock:
                state = {k: np.copy(v) if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v) for k, v in shared_mpc_state.items()}

            if SYNC_MPC and state["t_snap"] <= last_solve_t: # type: ignore
                time.sleep(0.001)
                continue
                
            try:
                v0 = np.concatenate([state["v_act"], state["w_act"], state["vj_act"]])
                stance_arr = np.array(state["horizon"]["stance"]).T # type: ignore
                swing_vz = np.array(state["horizon"]["swing_vz"]).T # type: ignore
                vz_active = np.array(state["horizon"]["vz_active"]).T # type: ignore
                fc_guess = np.array(state["horizon"]["fc_guess"]).reshape(mpc.n, 12).T # type: ignore

                # Directly use the corrected velocities passed from the main loop
                v_base_des = np.array([
                    state["target_vx"], 
                    state["target_vy"], 
                    0.0, 0.0, 0.0, 
                    state["yaw_rate"]
                ])

                yaw = float(state["rpy_act"][2])
                R_des = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                                  [np.sin(yaw),  np.cos(yaw), 0.0],
                                  [0.0,          0.0,         1.0]])

                tau_traj, qj_traj, v_traj = mpc.solve(
                    p0=state["p_act"],
                    R0=state["R_act"],
                    qj0=state["qj_act"],
                    v0=v0,
                    z_des=z_nom,
                    R_des=R_des,
                    q_nom=NOMINAL_STANCE,
                    v_base_des=v_base_des,
                    swing_vz_schedule=swing_vz,
                    stance_schedule=stance_arr,
                    mpc_weights=shared_mpc_weights,
                    vz_active_schedule=vz_active,
                    fc_guess=fc_guess,
                )
                # print(f"[mpc_worker] solve returned", flush=True)
                solve_summary(state["t_snap"], stance_arr, "ACPT" if mpc.last_solve_accepted else ("OK  " if mpc.last_solve_fresh else "SKIP"))

                if mpc.last_solve_fresh:
                    with mpc_lock:
                        mpc_output["tau_traj"] = tau_traj
                        mpc_output["qj_traj"] = qj_traj
                        mpc_output["v_traj"] = v_traj
                        mpc_output["t0"] = state["t_snap"]
                    last_solve_t = state["t_snap"]

                now = time.time()
                mpc_debug["solve_count"] += 1
                mpc_debug["last_solve_t"] = now
                mpc_debug["last_exc"] = None
            except Exception as e:
                mpc_debug["last_exc"] = repr(e)
                solve_summary(
                    state["t_snap"],
                    np.array(state["horizon"]["stance"]).T, # type: ignore[index] 
                    "FAIL" 
                )
                last_solve_t = state["t_snap"]
                with mpc_lock:
                    mpc_output["failed_t"] = state["t_snap"]
                
            # Enforce the ~50Hz control rate
            elapsed = time.time() - start_t
            if elapsed < mpc.dt:
                time.sleep(mpc.dt - elapsed)

    # Start the background solver
    threading.Thread(target=mpc_worker, daemon=True).start()
    continous_yaw = 0.0
    prev_raw_yaw = 0.0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                q_act = np.zeros(12)
                qd_act = np.zeros(12)
                idx = 0
                for leg in LEG_NAMES:
                    for j in JOINT_NAMES[leg]:
                        q_act[idx] = data.qpos[joint_info[j]['qpos_adr']]
                        qd_act[idx] = data.qvel[joint_info[j]['qvel_adr']]
                        idx += 1

                if logic.turning and not was_turning:
                    turn_anchor_xy = data.qpos[0:2].copy()
                was_turning = logic.turning

                if step_counter % decimation_steps == 0:
                    q_sorted = np.zeros(12)
                    qd_sorted = np.zeros(12)
                    idx = 0
                    for leg in LEG_NAMES:
                        for j in JOINT_NAMES[leg]:
                            q_sorted[idx] = data.qpos[joint_info[j]['qpos_adr']]
                            qd_sorted[idx] = data.qvel[joint_info[j]['qvel_adr']]
                            idx += 1
                    
                    logic.current_q = q_sorted
                    logic.current_q_dot = qd_sorted
                    
                    qw, qx, qy, qz = data.qpos[3:7]
                    sinr_cosp = 2.0 * (qw * qx + qy * qz)
                    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                    logic.current_roll = m.atan2(sinr_cosp, cosr_cosp)
                    
                    siny_cosp = 2.0 * (qw * qz + qx * qy)
                    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                    raw_yaw = m.atan2(siny_cosp, cosy_cosp)

                    delta_yaw = raw_yaw - prev_raw_yaw
                    delta_yaw = (delta_yaw + m.pi) % (2.0 * m.pi) - m.pi

                    continous_yaw += delta_yaw
                    prev_raw_yaw = raw_yaw
                    logic.current_yaw = continous_yaw

                    logic.loop_step(data.time)

                if step_counter % mpc_decimation == 0:
                    # Update variables
                    p_act = data.qpos[0:3].copy()
                    v_act = data.qvel[0:3].copy()
                    w_act = data.qvel[3:6].copy()
                    
                    qw, qx, qy, qz = data.qpos[3:7]
                    sinp = 2.0 * (qw * qy - qz * qx)
                    current_pitch = m.asin(np.clip(sinp, -1.0, 1.0))
                    rpy_act = np.array([logic.current_roll, current_pitch, logic.current_yaw])
                    foot_positions = [data.xpos[fid].copy() for fid in foot_ids]

                    R_act = np.zeros(9)
                    mujoco.mju_quat2Mat(R_act, data.qpos[3:7])
                    R_act = R_act.reshape(3, 3)

                    horizon = logic.mpc_horizon(data.time, mpc.n, mpc.dt)
                    w_act = R_act @ w_act

                    target_vy = 0.0
                    target_vx = 0.0
                    if logic.walking and isinstance(logic.planner, GaitPlanner):
                        gp = logic.planner.p
                        active_step_len = gp.step_len * logic.planner.stride_scale
                        target_vy = -((active_step_len * gp.freq) / gp.duty)
                    elif logic.turning:
                        kp_pos = 1.5
                        target_vx = np.clip(kp_pos * (turn_anchor_xy[0] - p_act[0]), -0.2, 0.2)
                        target_vy = np.clip(kp_pos * (turn_anchor_xy[1] - p_act[1]), -0.2, 0.2)
                    
                    # Mailbox Drop-off: Hand data to the background thread
                    with mpc_lock:
                        shared_mpc_state["p_act"] = p_act
                        shared_mpc_state["t_snap"] = data.time
                        shared_mpc_state["R_act"] = R_act
                        shared_mpc_state["v_act"] = v_act
                        shared_mpc_state["rpy_act"] = rpy_act
                        shared_mpc_state["w_act"] = w_act
                        shared_mpc_state["qj_act"] = q_act
                        shared_mpc_state["vj_act"] = qd_act
                        shared_mpc_state["q_des"] = cmd["q_des"]
                        shared_mpc_state["qd_des"] = cmd["qd_des"]
                        shared_mpc_state["foot_positions"] = foot_positions
                        shared_mpc_state["horizon"] = horizon
                        shared_mpc_state["z_off"] = logic.z_off
                        shared_mpc_state["target_vx"] = target_vx
                        shared_mpc_state["target_vy"] = target_vy
                        shared_mpc_state["target_yaw"] = logic.target_yaw
                        shared_mpc_state["yaw_rate"] = logic.yaw_rate
                        if logic.walking or logic.current_state == "RUN":
                            shared_mpc_state["p_ref_xy"][0] = p_act[0]
                            shared_mpc_state["p_ref_xy"][1] += target_vy * mpc.dt 
                        else:
                            shared_mpc_state["p_ref_xy"] = turn_anchor_xy if logic.turning else p_act[0:2]

                    if SYNC_MPC:
                        snap_t = data.time
                        while mpc_running:
                            with mpc_lock:
                                if mpc_output["t0"] >= snap_t or mpc_output["failed_t"] >= snap_t:
                                    break
                            time.sleep(0.001)

                with mpc_lock:
                    tau_traj = np.copy(mpc_output["tau_traj"])
                    qj_traj = np.copy(mpc_output["qj_traj"])
                    v_traj = np.copy(mpc_output["v_traj"])
                    t0 = mpc_output["t0"]

                k_f = np.clip((data.time - t0) / mpc.dt, 0.0, mpc.n)
                k0 = int(np.floor(k_f))
                k1 = min(k0 + 1, mpc.n)
                frac = k_f - k0

                q_des_mpc = qj_traj[:, k0] * (1 - frac) + qj_traj[:, k1] * frac
                qd_des_mpc = v_traj[:, k0] * (1 - frac) + v_traj[:, k1] * frac
                tau0 = tau_traj[:, min(k0, mpc.n - 1)]
                tau0 = np.nan_to_num(tau0, nan=0.0, posinf=1500.0, neginf=-1500.0)

                pd_torques_arr = fosmc.compute(
                    q=q_act,
                    q_dot=qd_act,
                    q_d=q_des_mpc,
                    q_dot_d=qd_des_mpc
                )
                pd_torques_arr = np.nan_to_num(pd_torques_arr, nan=0.0, posinf=1500.0, neginf=-1500.0)

                idx = 0
                applied_torque = np.zeros(12)
                for leg in LEG_NAMES:
                    for j in JOINT_NAMES[leg]:
                        info = joint_info[j]
                        final_torque = tau0[idx] + (pd_torques_arr[idx] if USE_FOSMC else 0.0)
                        clipped = np.clip(final_torque, -1500.0, 1500.0)
                        data.ctrl[info['actuator_id']] = clipped
                        applied_torque[idx] = clipped
                        idx += 1

                # --- TORQUE / SOLVE-RATE DEBUG ---
                if step_counter % (decimation_steps * 25) == 0:  # twice a second
                    print("\n=== MPC/FOSMC DEBUG ===")
                    print(f"MPC solves so far: {mpc_debug['solve_count']} | "
                          f"time since last solve: {time.time() - mpc_debug['last_solve_t']:.2f}s | "
                          f"last exception: {mpc_debug['last_exc']}")
                    print(f"GaitLogic state: {logic.current_state} | walking={logic.walking} turning={logic.turning} jump_state='{logic.jump_state}'")
                    print(f"cmd['q_des'] (raw gait target, NOT what the MPC tracks anymore): {np.round(cmd['q_des'], 3)}")
                    print(f"cmd['is_stance']              : {cmd['is_stance']}")
                    print(f"q_act        : {np.round(q_act, 3)}")
                    print(f"q_des (mpc)  : {np.round(q_des_mpc, 3)}   <- converges to NOMINAL_STANCE {NOMINAL_STANCE}, not cmd['q_des']")
                    print(f"q plan end   : {np.round(qj_traj[:, -1], 3)}   <- where the MPC sends the joints by node n")
                    with mpc_lock:
                        _target_vy = shared_mpc_state["target_vy"]
                    print(f"base v_y actual: {data.qvel[1]:.4f}   base v_y commanded: {_target_vy:.4f}")
                    print(f"tau0 (mpc ff): {np.round(tau0, 2)}")
                    print(f"pd (fosmc fb): {np.round(pd_torques_arr, 2)}")
                    print(f"applied ctrl : {np.round(applied_torque, 2)}")
                    print(f"base pos     : {np.round(data.qpos[0:3], 4)}  base quat: {np.round(data.qpos[3:7], 4)}")
                    R_dbg = np.zeros(9)
                    mujoco.mju_quat2Mat(R_dbg, data.qpos[3:7])
                    R_dbg = R_dbg.reshape(3, 3)
                    qw_, qx_, qy_, qz_ = data.qpos[3:7]
                    pitch_dbg = m.degrees(m.atan2(2.0 * (qw_ * qx_ + qy_ * qz_), 1.0 - 2.0 * (qx_ * qx_ + qy_ * qy_)))
                    foot_y_rel = [float((R_dbg.T @ (data.xpos[f] - data.xpos[h]))[1]) for f, h in zip(foot_ids, hip_ids)]
                    print(f"pitch (deg, + = nose down): {pitch_dbg:.2f} | foot y rel hip (+ = behind) FL FR BL BR: {np.round(foot_y_rel, 2)}")
                    print(f"swing-phase floor contact (2 ms steps since last print) FL FR BL BR: {scuff_steps}")
                    scuff_steps[:] = 0
                    print("=======================\n")

                mujoco.mj_step(model, data)
                # DEBUG: floor contact while the schedule says the foot is swinging
                if data.ncon > 0:
                    contact_bodies = np.concatenate([model.geom_bodyid[data.contact.geom1], model.geom_bodyid[data.contact.geom2]])
                    touching = np.isin(foot_ids, contact_bodies)
                    for i in range(4):
                        if touching[i] and not cmd["is_stance"][i]:
                            scuff_steps[i] += 1
                mujoco.mj_step(model, data)
                current_time = time.time()
                if (current_time - last_render_time) > 0.016:
                    viewer.sync()
                    last_render_time = current_time
                
                if step_counter % record_steps == 0:
                    mujoco.mjv_updateScene(model, data, viewer.opt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
                    video_writer.append_data(renderer.render())

                step_counter += 1

                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)
                    
    finally:
        video_writer.close()
        mpc_running = False 

if __name__ == '__main__':
    main()
