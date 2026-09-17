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

from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.FOSMCLogic import FOSMC
from LOGIC.MpcLogic import CentroidalMPC
from LOGIC.DynamicsLogic import QuadrupedDynamics
from ROS.BaseGUI import GUI
from pathlib import Path

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

    sensor_names = ['tl_foot_force_sensor', 'tr_foot_force_sensor', 'bl_foot_force_sensor', 'br_foot_force_sensor']
    sensor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in sensor_names]
    sensor_adr = [model.sensor_adr[sid] for sid in sensor_ids]
    # ---------------- 2. Setup Logic & UI ----------------
    
    cmd = {
        "q_des": np.zeros(12),
        "qd_des": np.zeros(12),
        "qdd_des": np.zeros(12),
        "foot_forces": np.zeros((4, 3)),
        "is_stance": [False, False, False, False],
        "kp": 600.0,
        "kd": 25.0,
        "swing_fraction": [0.0, 0.0, 0.0, 0.0]
    }

    graph_queue = queue.Queue()

    def handle_walk_points(points_data):
        if not points_data: return
        pt = points_data[0] 
        cmd["q_des"] = np.array(pt["positions"])
        cmd["qd_des"] = np.array(pt["velocities"])
        cmd["qdd_des"] = np.array(pt["accelerations"])
        cmd["is_stance"] = pt["is_stance"]
        cmd["foot_forces"] = pt["foot_forces"]
        cmd["swing_fraction"] = pt.get("swing_fraction", [0.0, 0.0, 0.0, 0.0])

    def handle_jump_points(q_desired, qd_desired, qdd_desired, foot_forces, is_stance):
        cmd["q_des"] = np.array(q_desired)
        cmd["qd_des"] = np.array(qd_desired)
        cmd["qdd_des"] = np.array(qdd_desired)
        cmd["foot_forces"] = np.array(foot_forces)
        cmd["is_stance"] = is_stance

    def handle_transition(current_angles):
        cmd["q_des"] = np.array(current_angles)
        cmd["qd_des"] = np.zeros(12)
        cmd["qdd_des"] = np.zeros(12)
        
    def handle_raw_tune(raw_angles):
        coxa, femur, tibia = raw_angles
        positions = []
        for _ in range(4):
            positions.extend([coxa, femur, tibia])
        cmd["q_des"] = np.array(positions)
        cmd["qd_des"] = np.zeros(12)
        cmd["qdd_des"] = np.zeros(12)
        
    def handle_graph_push(graph_data):
        graph_queue.put(graph_data)

    logic = GaitLogic({
        "walk_points": handle_walk_points,
        "jump_points": handle_jump_points,
        "transition_cb": handle_transition,
        "raw_tune_cb": handle_raw_tune,
        "graph": handle_graph_push
    })

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
        Ke1=24.0,
        Ke2=5.0,
        Ks=25.0,
        Kr=10.0,
        gamma_c=0.01,
        gamma_a=0.01
    )

    record_hz = 60
    record_steps = int(physics_hz / record_hz)
    renderer = mujoco.Renderer(model, height=480, width=640)
    video_writer = imageio.get_writer('simulation.mp4', fps=record_hz)

    last_render_time = time.time()

    dyn = QuadrupedDynamics(urdf_path)
    mpc = CentroidalMPC(urdf_path, n=10, dt=0.02)

    physics_hz = int(1.0 / model.opt.timestep)
    mpc_hz = 50
    mpc_decimation = int(physics_hz / mpc_hz)


    # ---------------- 4. Asynchronous MPC Setup ----------------
    mpc_lock = threading.Lock()
    
    # Shared state dictionary for the MPC thread to read from
    shared_mpc_state = {
        "p_act": np.zeros(3),
        "v_act": np.zeros(3),
        "rpy_act": np.zeros(3),
        "w_act": np.zeros(3),
        "foot_positions": [np.zeros(3) for _ in range(4)],
        "stance_schedule": [[True, True, True, True] for _ in range(mpc.n)],
        "z_off": 2.5,
        "target_yaw": 0.0,
        "yaw_rate": 0.0,
        "p_ref_xy": np.zeros(2)
    }
    
    # This will hold the latest solved forces for the physics loop to consume
    global optimal_foot_forces
    optimal_foot_forces = np.zeros(12)
    planned_stance_schedule = []
    turn_anchor_xy = np.zeros(2)
    was_turning = False
    mpc_running = True

    foot_contact_fz = np.zeros(4)
    leg_touched_down = [False, False, False, False]
    fz_alpha = 0.3
    contact_force_threshold = 2.0

    # Continuous per-leg handoff weight (0 = pure FOSMC swing torque,
    # 1 = pure stance torque). Ramped instead of snapped so the
    # FOSMC -> MPC/PD switch doesn't inject a torque step at touchdown.
    contact_blend = np.zeros(4)
    CONTACT_BLEND_TIME = 0.04  # seconds to fully ramp across a footfall event

    def mpc_worker():
        global optimal_foot_forces
        while mpc_running:
            start_t = time.time()
            
            # Safely copy current state
            with mpc_lock:
                state = {k: np.copy(v) if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v) for k, v in shared_mpc_state.items()}
                
            try:
                # Calculate required angular velocity for the MPC so it doesn't fight the turn
                forces = mpc.solve(
                    p_init=state["p_act"],
                    v_init=state["v_act"],
                    rpy_init=state["rpy_act"],
                    w_init=state["w_act"],
                    foot_positions=state["foot_positions"],
                    stance_schedule=state["stance_schedule"],
                    p_ref=np.array([state["p_ref_xy"][0], state["p_ref_xy"][1], float(state["z_off"]) - 2.5]), # type: ignore[arg-type] 
                    v_ref=np.array([0.0, state["target_vy"], 0.0]),
                    rpy_ref=np.array([state["rpy_act"][0], state["rpy_act"][1], state["target_yaw"]]),
                    w_ref=np.array([0.0, 0.0, state["yaw_rate"]])
                )
                # Safely update the forces available to the physics engine
                with mpc_lock:
                    optimal_foot_forces = forces
            except Exception as e:
                # Silently pass to avoid spamming the console during edge-case solver failures
                pass
                
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
                    # logic.current_yaw = m.atan2(siny_cosp, cosy_cosp)

                    # # --- FACING DIRECTION DEBUG ---
                    # if step_counter % (decimation_steps * 25) == 0:
                    #     mat = np.zeros(9)
                    #     mujoco.mju_quat2Mat(mat, data.qpos[3:7])
                    #     mat = mat.reshape(3, 3)
                    #
                    #     # Assuming local +Y is the front face of your new URDF
                    #     local_front = np.array([0.0, 1.0, 0.0])
                    #     global_front = mat @ local_front
                    #
                    #     print(f"\n=== FACING DIRECTION ===")
                    #     print(f"Global Front Vector -> X: {global_front[0]:.3f}, Y: {global_front[1]:.3f}, Z: {global_front[2]:.3f}")
                    #     print(f"Current Yaw (rad): {logic.current_yaw:.3f}")
                    #     print("========================\n")
                    # # ------------------------------

                    logic.loop_step()
                    raw_fz = np.array([
                        np.linalg.norm(data.sensordata[sensor_adr[i]:sensor_adr[i] + 3])
                        for i in range(4)
                    ])
                    # foot_contact_fz = (1.0 - fz_alpha) * foot_contact_fz + fz_alpha * raw_fz
                    for i in range(4):
                        if not cmd["is_stance"][i]:
                            leg_touched_down[i] = False
                        elif raw_fz[i] > contact_force_threshold:
                            leg_touched_down[i] = True

                    for i in range(4):
                        if not logic.walking and not logic.turning and logic.jump_state == "":
                            # Idle standing: trust the schedule directly (unchanged behavior)
                            target = 1.0
                        elif logic.walking or logic.turning:
                            target = 1.0 if leg_touched_down[i] else 0.0
                        else:
                            # Jumping: unchanged behavior, no contact gating
                            target = 1.0 if cmd["is_stance"][i] else 0.0

                        step = logic.dt / CONTACT_BLEND_TIME
                        if contact_blend[i] < target:
                            contact_blend[i] = min(target, contact_blend[i] + step)
                        else:
                            contact_blend[i] = max(target, contact_blend[i] - step)

                # # --- SKELETON VERIFICATION DEBUG ---
                # if step_counter % (decimation_steps * 25) == 0:  # Triggers twice a second
                #     print("\n=== SKELETON ALIGNMENT VERIFICATION ===")
                #
                #     # 1. FOSMC's Analytical Skeleton (KinematicsLogic FK)
                #     # Grabbing the current angles for the Front Left (FL) leg[cite: 44]
                #     t1 = m.degrees(logic.current_q[0])
                #     t2 = m.degrees(logic.current_q[1])
                #     t3 = m.degrees(logic.current_q[2])
                #
                #     # Calculating where the math thinks the foot is[cite: 40]
                #     analytical_fl = logic.kinematics.fk('FL', t1, t2, t3)
                #     fosmc_pos = np.array([analytical_fl[0, 3], analytical_fl[1, 3], analytical_fl[2, 3]])
                #
                #     # 2. MPC's Physical Skeleton (MuJoCo/Pinocchio)
                #     # Grabbing the absolute world coordinates of the foot and the base[cite: 44]
                #     fl_foot_world = data.xpos[foot_ids[0]]
                #     base_world = data.qpos[0:3]
                #
                #     # Finding the physical foot position relative to the body center
                #     mpc_pos = fl_foot_world - base_world
                #
                #     print(f"FOSMC (Analytical FL) -> X: {fosmc_pos[0]:.3f}, Y: {fosmc_pos[1]:.3f}, Z: {fosmc_pos[2]:.3f}")
                #     print(f"MPC   (Physical FL)   -> X: {mpc_pos[0]:.3f}, Y: {mpc_pos[1]:.3f}, Z: {mpc_pos[2]:.3f}")
                #     print("=======================================\n")
                # # -----------------------------------

                if step_counter % mpc_decimation == 0:
                    # Update variables
                    p_act = data.qpos[0:3]
                    v_act = data.qvel[0:3]
                    w_act = data.qvel[3:6]
                    
                    qw, qx, qy, qz = data.qpos[3:7]
                    sinp = 2.0 * (qw * qy - qz * qx)
                    current_pitch = m.asin(np.clip(sinp, -1.0, 1.0))
                    rpy_act = np.array([logic.current_roll, current_pitch, logic.current_yaw])
                    foot_positions = [data.xpos[fid] for fid in foot_ids]

                    planned_stance_schedule = []
                    if not logic.walking and not logic.turning and logic.jump_state == "":
                        planned_stance_schedule = [[True, True, True, True] for _ in range(mpc.n)]
                    else:
                        omega = 2.0 * m.pi * logic.gait_freq
                        stance_limit = 2.0 * m.pi * logic.duty_factor
                        for k in range(mpc.n):
                            t_future = logic.t + (k * mpc.dt)
                            phase_future = (omega * t_future) % (2.0 * m.pi)
                            k_stance = [((phase_future + logic.phase_offsets[leg]) % (2.0 * m.pi)) < stance_limit for leg in LEG_NAMES]
                            if k == 0:
                                # "Right now" is measurable. Don't let the QP budget
                                # weight onto a foot that hasn't actually landed —
                                # future ticks stay on the forecast, since contact
                                # there can't be sensed in advance.
                                k_stance = [k_stance[i] and leg_touched_down[i] for i in range(4)]
                            planned_stance_schedule.append(k_stance)

                    target_vy = 0.0
                    if logic.walking or logic.current_state == "RUN":
                        ramp_duration = 1.0
                        ramp_factor = min(logic.t / ramp_duration, 1.0)
                        active_step_len = logic.step_len * ramp_factor
                        target_vy = -((active_step_len * logic.gait_freq) / logic.duty_factor)
                    
                    # Mailbox Drop-off: Hand data to the background thread
                    with mpc_lock:
                        shared_mpc_state["p_act"] = p_act
                        shared_mpc_state["v_act"] = v_act
                        shared_mpc_state["rpy_act"] = rpy_act
                        shared_mpc_state["w_act"] = w_act
                        shared_mpc_state["foot_positions"] = foot_positions
                        shared_mpc_state["stance_schedule"] = planned_stance_schedule
                        shared_mpc_state["z_off"] = logic.z_off
                        shared_mpc_state["target_vy"] = target_vy
                        shared_mpc_state["target_yaw"] = logic.target_yaw
                        shared_mpc_state["yaw_rate"] = logic.yaw_rate
                        if logic.walking or logic.current_state == "RUN":
                            shared_mpc_state["p_ref_xy"][0] = p_act[0]
                            shared_mpc_state["p_ref_xy"][1] += target_vy * mpc.dt 
                        else:
                            shared_mpc_state["p_ref_xy"] = turn_anchor_xy if logic.turning else p_act[0:2]


                q_act = np.zeros(12)
                qd_act = np.zeros(12)
                idx = 0
                for leg in LEG_NAMES:
                    for j in JOINT_NAMES[leg]:
                        q_act[idx] = data.qpos[joint_info[j]['qpos_adr']]
                        qd_act[idx] = data.qvel[joint_info[j]['qvel_adr']]
                        idx += 1

                qdd_des_full = np.zeros(model.nv)
                pd_torques = np.zeros(model.nv)
                
                q_des_interp_arr = np.zeros(12)
                qd_des_interp_arr = np.zeros(12)
                q_act_arr = np.zeros(12)
                qd_act_arr = np.zeros(12)
                adr_map = []

                idx = 0
                for leg in LEG_NAMES:
                    for j in JOINT_NAMES[leg]:
                        adr = joint_info[j]['qvel_adr']
                        adr_map.append(adr)
                        
                        qdd_des_full[adr] = cmd["qdd_des"][idx]
                        
                        dt_sub = (step_counter % decimation_steps) * model.opt.timestep
                        q_des_interp_arr[idx] = cmd["q_des"][idx] + (cmd["qd_des"][idx] * dt_sub)
                        qd_des_interp_arr[idx] = cmd["qd_des"][idx] + (cmd["qdd_des"][idx] * dt_sub)
                        
                        q_act_arr[idx] = q_act[idx]
                        qd_act_arr[idx] = qd_act[idx]
                        
                        idx += 1

                pd_torques_arr = fosmc.compute(
                    q=q_act_arr, 
                    q_dot=qd_act_arr, 
                    q_d=q_des_interp_arr, 
                    q_dot_d=qd_des_interp_arr
                )

                pd_torques_arr = np.nan_to_num(pd_torques_arr, nan=0.0, posinf=1500.0, neginf=-1500.0)

                for i, adr in enumerate(adr_map):
                    pd_torques[adr] = float(pd_torques_arr[i])
                        
                tau_ff = np.zeros(model.nv)
                q_pin = np.zeros(dyn.model.nq)
                q_pin[0:3] = data.qpos[0:3]
                q_pin[3:7] = [
                    data.qpos[4],
                    data.qpos[5],
                    data.qpos[6],
                    data.qpos[3],
                ]  # Pinocchio uses [x,y,z,w]

                # Accurately map joint positions using Pinocchio IDs to avoid URDF order mismatch
                for leg in LEG_NAMES:
                    for j in JOINT_NAMES[leg]:
                        pin_id = dyn.model.getJointId(j)
                        idx_q = dyn.model.joints[pin_id].idx_q
                        q_pin[idx_q] = data.qpos[joint_info[j]['qpos_adr']]

                jacobians = dyn.get_foot_jacobians(q_pin)

                for i in range(4):
                    F_i = optimal_foot_forces[3 * i : 3 * i + 3]
                    # Extract only the 12 actuated joints (columns 6 to 18)
                    J_joints = jacobians[i][:, 6:]
                    tau_joints_pin = -J_joints.T @ F_i

                    # Accurately map torques back to MuJoCo using Pinocchio IDs
                    for leg in LEG_NAMES:
                        for j in JOINT_NAMES[leg]:
                            pin_id = dyn.model.getJointId(j)
                            idx_v = dyn.model.joints[pin_id].idx_v - 6
                            adr = joint_info[j]['qvel_adr']
                            tau_ff[adr] += tau_joints_pin[idx_v]

                idx = 0
                for i, leg in enumerate(LEG_NAMES):
                    w = contact_blend[i]  # 0 = full swing (FOSMC), 1 = full stance (MPC+PD)
                    for j in JOINT_NAMES[leg]:
                        info = joint_info[j]
                        adr = info['qvel_adr']

                        pos_err = q_des_interp_arr[idx] - q_act_arr[idx]
                        vel_err = qd_des_interp_arr[idx] - qd_act_arr[idx]
                        pd_correction = cmd["kp"] * pos_err + cmd["kd"] * vel_err
                        SWING_TAPER_START = 0.85
                        SWING_TAPER_MIN = 0.35
                        sf = cmd["swing_fraction"][i]
                        if sf > SWING_TAPER_START:
                            taper = 1.0 - (1.0 - SWING_TAPER_MIN) * (sf - SWING_TAPER_START) / (1.0 - SWING_TAPER_START)
                        else:
                            taper = 1.0
                        swing_torque = pd_torques[adr] * taper

                        final_torque = (1.0 - w) * swing_torque + w * tau_ff[adr] + (w ** 2) * pd_correction
                        data.ctrl[info['actuator_id']] = np.clip(final_torque, -1500.0, 1500.0)
                        idx += 1

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
