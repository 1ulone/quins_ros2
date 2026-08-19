import re
import os
import sys
import time
import queue
import mujoco
import tempfile
import threading
import mujoco.viewer
import math as m
import numpy as np
import xml.etree.ElementTree as ET

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import your new framework-agnostic architecture
from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from ROS.BaseGUI import GUI

def main():
    # ---------------- 1. Setup Mujoco Environment ----------------
    urdf_path = '/home/ulone/ros2_ws/src/quins/urdf/quadruped.urdf'
    absolute_pkg_path = '/home/ulone/ros2_ws/src/quins/'

    with open(urdf_path, 'r') as file:
        urdf_xml = file.read()

    urdf_xml = urdf_xml.replace('package://quins/', absolute_pkg_path)
    urdf_xml = re.sub(r'<xacro:arg.*?>', '', urdf_xml)
    urdf_xml = re.sub(r'(<robot[^>]*>)', r'\1\n<mujoco><compiler fusestatic="false"/></mujoco>', urdf_xml, count=1)

    temp_model = mujoco.MjModel.from_xml_string(urdf_xml)
    temp_mjcf = tempfile.NamedTemporaryFile(delete=False, suffix='.xml')
    temp_mjcf.close()
    mujoco.mj_saveLastXML(temp_mjcf.name, temp_model)

    with open(temp_mjcf.name, 'r') as file:
        mjcf_xml = file.read()

    environment_injection = """
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="32" height="32"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" directional="true"/>
        <geom name="floor" type="plane" pos="0 0 -2.5" size="100 100 0.1" material="grid" condim="3" friction="1.0 0.005 0.0001"/>
    """
    mjcf_xml = mjcf_xml.replace('<worldbody>', environment_injection)

    actuators_xml = "<actuator>\n"
    for leg, joints in JOINT_NAMES.items():
        for joint in joints:
            actuators_xml += f'    <motor name="{joint}_motor" joint="{joint}" gear="1" ctrllimited="true" ctrlrange="-1500 1500"/>\n'
    actuators_xml += "</actuator>\n"
    
    # Inject default joint damping to match Gazebo's ODE solver
    damping_xml = "<default>\n    <joint damping=\"0.05\" frictionloss=\"0.01\"/>\n</default>\n"
    mjcf_xml = mjcf_xml.replace('<worldbody>', f'{damping_xml}<worldbody>')
    mjcf_xml = mjcf_xml.replace('</worldbody>', f'</worldbody>\n{actuators_xml}')

    root_xml = ET.fromstring(mjcf_xml)
    worldbody = root_xml.find('worldbody')
    if worldbody is not None:
        for body in worldbody.findall('body'):
            if body.find('joint') is None:
                body.set('pos', '0 0 1.5')
                ET.SubElement(body, 'freejoint', name='root_floating_base')
                break
    mjcf_xml = ET.tostring(root_xml, encoding='unicode')

    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    print(f"Total MuJoCo Mass: {mujoco.mj_getTotalmass(model):.2f} kg")
    data = mujoco.MjData(model)

    model.geom_contype[:] = 1
    model.geom_conaffinity[:] = 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id != -1:
        model.geom_contype[floor_id] = 0
        model.geom_conaffinity[floor_id] = 1

    joint_info = {}
    for leg, joints in JOINT_NAMES.items():
        for joint in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            joint_info[joint] = {
                'qpos_adr': model.jnt_qposadr[jid],
                'qvel_adr': model.jnt_dofadr[jid],
                'actuator_id': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_motor"),
            }

    foot_body_names = ['fl_foot', 'fr_foot', 'rl_foot', 'rr_foot']
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names]

    # ---------------- 2. Setup Logic & UI ----------------
    
    # Command buffer updated by GaitLogic callbacks
    cmd = {
        "q_des": np.zeros(12),
        "qd_des": np.zeros(12),
        "qdd_des": np.zeros(12),
        "foot_forces": np.zeros((4, 3)),
        "is_stance": [False, False, False, False],
        "kp": 600.0,
        "kd": 25.0
    }

    # Create a thread-safe pipeline for the graph data
    graph_queue = queue.Queue()

    def handle_walk_points(points_data):
        if not points_data: return
        pt = points_data[0] 
        cmd["q_des"] = np.array(pt["positions"])
        cmd["qd_des"] = np.array(pt["velocities"])
        cmd["qdd_des"] = np.array(pt["accelerations"])
        cmd["is_stance"] = pt["is_stance"]
        cmd["foot_forces"] = pt["foot_forces"]

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
        # MuJoCo Thread: Push data to the queue safely, don't draw!
        graph_queue.put(graph_data)

    # Initialize Logic
    logic = GaitLogic({
        "walk_points": handle_walk_points,
        "jump_points": handle_jump_points,
        "transition_cb": handle_transition,
        "raw_tune_cb": handle_raw_tune,
        "graph": handle_graph_push  # Assign the safe queue function
    })

    # Initialize GUI in its own thread
    def start_gui():
        gui = GUI({
            "state": logic.update_state,
            "phase": logic.update_phase_offsets,
            "wt_params": logic.update_wt_params,
            "jt_params": logic.update_jt_params,
            "raw_tune": logic.raw_tune
        })
        gui.setup()
        
        while True:
            needs_redraw = False
            
            # GUI Thread: Drain the queue and append data without drawing yet
            while not graph_queue.empty():
                g_data = graph_queue.get_nowait()
                gui.time_history.append(g_data[0])
                gui.desired_history.append(g_data[1])
                gui.measured_history.append(g_data[2])
                needs_redraw = True
                
            # Only command Matplotlib to draw ONCE per UI loop to prevent lag
            if needs_redraw:
                gui.refresh_graph()
                
            gui.update()
            time.sleep(0.05)

    threading.Thread(target=start_gui, daemon=True).start()

    # ---------------- 3. Simulation Loop ----------------
    physics_hz = 1.0 / model.opt.timestep # usually 500Hz
    control_hz = logic.control_rate       # matches 50.0Hz
    decimation_steps = int(physics_hz / control_hz)
    step_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- HIGH-LEVEL LOGIC (Runs at 50Hz) ---
            if step_counter % decimation_steps == 0:
                
                # A. Read Low-Freq Telemetry for GaitLogic
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
                logic.current_yaw = m.atan2(siny_cosp, cosy_cosp)

                # B. Execute Mathematical Core
                logic.loop_step()

            # --- LOW-LEVEL PD CONTROLLER (Runs at 500Hz Physics Rate) ---
            
            # 1. Read High-Freq Joint States for smooth PD calculation
            q_act = np.zeros(12)
            qd_act = np.zeros(12)
            idx = 0
            for leg in LEG_NAMES:
                for j in JOINT_NAMES[leg]:
                    q_act[idx] = data.qpos[joint_info[j]['qpos_adr']]
                    qd_act[idx] = data.qvel[joint_info[j]['qvel_adr']]
                    idx += 1

            # 2. Solve Mass Matrix and Bias Forces
            _m = np.zeros((model.nv, model.nv), dtype=np.float64)
            mujoco.mj_fullM(model, data, _m) 

            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            bias_forces = data.qfrc_bias

            # 3. Calculate Feedforward + PD Torques + GRF
            total_mass = mujoco.mj_getTotalmass(model)
            gravity = 9.81
            mg_half = (total_mass * gravity) / 2.0
            
            tau_grf = np.zeros(model.nv)
            planned_stance = cmd["is_stance"]
            planned_forces = cmd["foot_forces"]

            foot_body_names = ['fl_foot', 'fr_foot', 'rl_foot', 'rr_foot']
            for i, name in enumerate(foot_body_names):
                if planned_stance[i]:
                    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    jacp = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jacp, None, fid)
                    
                    # Inject dynamic thrust calculated by GaitLogic
                    f_z = planned_forces[i][2] 
                    tau_grf += jacp.T @ np.array([0.0, 0.0, f_z])
            
            # --- Front Axle Distribution ---
            # front_active = []
            # if planned_stance[0]: front_active.append('fl_foot')
            # if planned_stance[1]: front_active.append('fr_foot')
            #
            # if front_active:
            #     f_z_front = mg_half / len(front_active)
            #     for name in front_active:
            #         fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            #         jacp = np.zeros((3, model.nv))
            #         mujoco.mj_jacBody(model, data, jacp, None, fid)
            #         tau_grf += jacp.T @ np.array([0.0, 0.0, f_z_front])
            #
            # # --- Rear Axle Distribution ---
            # rear_active = []
            # if planned_stance[2]: rear_active.append('rl_foot')
            # if planned_stance[3]: rear_active.append('rr_foot')
            #
            # if rear_active:
            #     f_z_rear = mg_half / len(rear_active)
            #     for name in rear_active:
            #         fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            #         jacp = np.zeros((3, model.nv))
            #         mujoco.mj_jacBody(model, data, jacp, None, fid)
            #         tau_grf += jacp.T @ np.array([0.0, 0.0, f_z_rear])

            qdd_des_full = np.zeros(model.nv)
            pd_torques = np.zeros(model.nv)
            
            idx = 0
            for leg in LEG_NAMES:
                for j in JOINT_NAMES[leg]:
                    adr = joint_info[j]['qvel_adr']
                    
                    qdd_des_full[adr] = cmd["qdd_des"][idx]
                    
                    dt_sub = (step_counter % decimation_steps) * model.opt.timestep
                    q_des_interp = cmd["q_des"][idx] + (cmd["qd_des"][idx] * dt_sub)
                    qd_des_interp = cmd["qd_des"][idx] + (cmd["qdd_des"][idx] * dt_sub)
                    
                    pos_err = q_des_interp - q_act[idx]
                    vel_err = qd_des_interp - qd_act[idx]
                    pd_torques[adr] = (cmd["kp"] * pos_err) + (cmd["kd"] * vel_err)
                    
                    idx += 1
                    
            ff_torques = (_m @ qdd_des_full) + bias_forces - tau_grf

            # 4. Apply Final Torques
            for leg in LEG_NAMES:
                for j in JOINT_NAMES[leg]:
                    info = joint_info[j]
                    adr = info['qvel_adr']
                    
                    final_torque = ff_torques[adr] + pd_torques[adr]
                    data.ctrl[info['actuator_id']] = np.clip(final_torque, -1500.0, 1500.0)

            # --- STEP PHYSICS ---
            mujoco.mj_step(model, data)
            viewer.sync()
            step_counter += 1

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    main()
