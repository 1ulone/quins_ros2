import re
import time
import mujoco
import tempfile
import threading
import mujoco.viewer
import math as m
import numpy as np
import tkinter as tk
import xml.etree.ElementTree as ET
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from LOGIC.KinematicsLogic import KinematicsLogic

BODY_WIDTH = 1.3334   
BODY_LENGTH = 2.2924  

LEG_NAMES = [
    'FL',
    'FR',
    'BL',
    'BR'
]

FOOT_BODY_NAMES = [
    'bl_tip_link',
    'br_tip_link',
    'tl_tip_link',
    'tr_tip_link',
]

JOINT_NAMES = {
    'FL': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
    'FR': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
    'BL': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
    'BR': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint'],
}

PHASE_OFFSETS = {
    'FL': 0.0,
    'BR': m.pi / 2.0,
    'FR': m.pi,
    'BL': 3.0 * m.pi / 2.0,
}

STARTUP_DELAY = 10.0

class GaitParams:
    def __init__(self):
        self.gait_freq = 2.00
        self.x_off = 0.30
        self.z_off = 2.70
        self.step_len = 2.00
        self.step_h = 0.75
        self.kp = 500.0
        self.kd = 20.0
        self.kz = 0.05
        self.k_roll = 0.0
        self.k_pitch = 0.50
        self.k_yaw = 0.0
        self.kd_roll = 0.0
        self.kd_pitch = 0.10

def start_tuner_ui(gait: GaitParams):
    root = tk.Tk()
    root.title("Gait Tuner")

    def make_slider(label, attr, frm, to, resolution=0.01):
        tk.Label(root, text=label).pack()

        def on_change(val):
            setattr(gait, attr, float(val))

        scale = tk.Scale(root, from_=frm, to=to, resolution=resolution,
                          orient="horizontal", length=300,
                          command=on_change)
        scale.set(getattr(gait, attr))
        scale.pack()

    make_slider("Gait Frequency", "gait_freq", 0.1, 5.0)
    make_slider("X Offset", "x_off", -2.0, 2.0)
    make_slider("Z Offset (Ride Height)", "z_off", 0.0, 4.0)
    make_slider("Step Length", "step_len", 0.0, 10.0)
    make_slider("Step Height", "step_h", 0.0, 3.0)
    make_slider("KP", "kp", 0.0, 500.0)
    make_slider("KD", "kd", 0.0, 100.0)
    make_slider("Kz", "kz", 0.0, 2.0)
    make_slider("K Roll", "k_roll", 0.0, 5.0)
    make_slider("K Pitch", "k_pitch", 0.0, 5.0)
    make_slider("K Yaw", "k_yaw", 0.0, 5.0)
    make_slider("KD Roll", "kd_roll", 0.0, 2.0)
    make_slider("KD Pitch", "kd_pitch", 0.0, 2.0)

    worst_reach = tk.StringVar()
    max_reach = tk.StringVar()
    headroom = tk.StringVar()
        
    def reload_reach(event=None):
        wr, mr, hr = reach_headroom(gait, 1.5005, 1.54138, BODY_WIDTH, BODY_LENGTH)
        worst_reach.set(str(wr))
        max_reach.set(str(mr))
        headroom.set(str(hr))

    tk.Label(root, textvariable=worst_reach).pack()
    tk.Label(root, textvariable=max_reach).pack()
    tk.Label(root, textvariable=headroom).pack()

    tk.Button(root, text="Reload reach", command=reload_reach).pack()

    root.mainloop()

def reach_headroom(gait, l2, l3, body_width, body_length, max_tilt=m.radians(15), margin=0.85):
    max_reach = margin * (l2 + l3)

    roll_corr_max  = gait.k_roll  * max_tilt * (body_width  / 2.0)
    pitch_corr_max = gait.k_pitch * max_tilt * (body_length / 2.0)
    yaw_corr_max   = gait.k_yaw   * max_tilt

    vertical   = gait.z_off + roll_corr_max + pitch_corr_max
    horizontal = gait.x_off + gait.step_len / 2.0 + yaw_corr_max

    worst_reach = m.sqrt(horizontal**2 + vertical**2)
    return worst_reach, max_reach, max_reach - worst_reach

def trajectory_controller(phase, step_len, step_h):
    z = 0.0
    if phase < m.pi:
        fraction = phase / m.pi
        x = -(step_len / 2.0) + (fraction * step_len)
        y = 0.0
    else:
        fraction = (phase - m.pi) / m.pi
        x = (step_len / 2.0) - (fraction * step_len)
        y = step_h * m.sin(fraction * m.pi)
    return x, y, z

def get_jacobian_constraint(model, data, leg_names):
    jacobians = []

    for body_name in leg_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jacp = np.zeros((3, model.nv))

        mujoco.mj_jacBodyCom(model, data, jacp, None, body_id)
        jacobians.append(jacp)

    if not jacobians:
        return np.zeros((model, 0)), np.zeros((0, model.nv))

    C_T = np.vstack(jacobians)
    C = C_T.T
    return C, C_T

def main():
    urdf_path = '/home/ulone/ros2_ws/src/quins/urdf/quadruped.urdf'
    absolute_pkg_path = '/home/ulone/ros2_ws/src/quins/'

    with open(urdf_path, 'r') as file:
        urdf_xml = file.read()

    urdf_xml = urdf_xml.replace('package://quins/', absolute_pkg_path)
    urdf_xml = re.sub(r'<xacro:arg.*?>', '', urdf_xml)

    urdf_xml = re.sub(
        r'(<robot[^>]*>)',
        r'\1\n<mujoco><compiler fusestatic="false"/></mujoco>',
        urdf_xml,
        count=1
    )

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
            actuators_xml += f'    <motor name="{joint}_motor" joint="{joint}" gear="1" ctrllimited="true" ctrlrange="-100 100"/>\n'
    actuators_xml += "</actuator>\n"

    mjcf_xml = mjcf_xml.replace('</worldbody>', f'</worldbody>\n{actuators_xml}')

    root_xml = ET.fromstring(mjcf_xml)
    worldbody = root_xml.find('worldbody')
    assert worldbody is not None

    for body in worldbody.findall('body'):
        if body.find('joint') is None:
            body.set('pos', '0 0 1.5')
            ET.SubElement(body, 'freejoint', name='root_floating_base')
            break

    mjcf_xml = ET.tostring(root_xml, encoding='unicode')

    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    model.geom_contype[:] = 1
    model.geom_conaffinity[:] = 0

    FOOT_BODY_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_BODY_NAMES]

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

    k = KinematicsLogic()

    gait = GaitParams()
    threading.Thread(target=start_tuner_ui, args=(gait,), daemon=True).start()

    physics_hz = 1.0 / model.opt.timestep
    control_hz = 500.0
    decimation_steps = int(physics_hz / control_hz)
    step_counter = 0
    body_ids = [1]

    dt_control = 1.0 / control_hz
    ref_state = {
        joint: { 'pos': None, 'vel': 0.0, 'acc': 0.0 }
        for joints in JOINT_NAMES.values()
        for joint in joints
    }

    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = gait.z_off
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # Identity quaternion

    for leg in LEG_NAMES:
        ix, iy, iz = k.get_init_pos(leg)
        tx = ix + gait.x_off
        ty = gait.z_off
        tz = iz
        theta1, theta2, theta3 = k.ik(leg, tx, ty, tz)
        targets = [theta1, theta2, theta3]

        # 2. Overwrite the qpos vector with the bent targets
        for i, joint_name in enumerate(JOINT_NAMES[leg]):
            info = joint_info[joint_name]
            data.qpos[info['qpos_adr']] = targets[i]
            ref_state[joint_name]['pos'] = targets[i]

    mujoco.mj_forward(model, data)
    
    telemetry_log = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if step_counter % decimation_steps == 0:
                t = data.time

                if t < STARTUP_DELAY:
                    phase_now = m.pi
                else:
                    omega = 2.0 * m.pi * gait.gait_freq
                    phase_now = (omega * (t - STARTUP_DELAY)) % (2.0 * m.pi)

                actual_z = data.qpos[2]
                qw, qx, qy, qz = data.qpos[3:7]

                sinr_cosp = 2.0 * (qw * qx + qy * qz)
                cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                actual_roll = m.atan2(sinr_cosp, cosr_cosp)

                sinp = 2.0 * (qw * qy - qz * qx)
                actual_pitch = m.asin(np.clip(sinp, -1.0, 1.0))

                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                actual_yaw = m.atan2(siny_cosp, cosy_cosp)

                roll_rate = data.qvel[3]
                pitch_rate = data.qvel[4]

                max_tilt = m.radians(15)
                roll_clamped = max(-max_tilt, min(max_tilt, actual_roll))
                pitch_clamped = max(-max_tilt, min(max_tilt, actual_pitch))
                
                z_err = gait.kz * (gait.z_off - actual_z)
                roll_err = gait.k_roll * (0.0 - roll_clamped) - gait.kd_roll * roll_rate
                pitch_err = gait.k_pitch * (0.0 - pitch_clamped) - gait.kd_pitch * pitch_rate
                yaw_err = gait.k_yaw * (0.0 - actual_yaw)

                data.qacc[0:6] = 0.0

                q_ddot_cmd = np.zeros(model.nv)
                is_stance_list = []

                for leg_idx, leg in enumerate(LEG_NAMES):
                    leg_phase = (phase_now + PHASE_OFFSETS[leg]) % (2.0 * m.pi)

                    is_stance = (leg_phase >= m.pi)
                    is_stance_list.append(is_stance)

                    xl, yl, zl = trajectory_controller(leg_phase, gait.step_len, gait.step_h)
                    ix, iy, iz = k.get_init_pos(leg)

                    is_front = 1.0 if 'F' in leg else -1.0
                    is_left = 1.0 if 'L' in leg else -1.0

                    leg_corr = z_err + (-is_left * roll_err * (BODY_WIDTH / 2.0)) + (is_front * pitch_err * (BODY_LENGTH / 2.0))
                    stride_corr = is_left * yaw_err

                    tx = ix + xl + gait.x_off + stride_corr
                    ty = gait.z_off - yl + leg_corr
                    tz = iz + zl

                    theta1, theta2, theta3 = k.ik(leg, tx, ty, tz)
                    if abs(theta1) > m.radians(45):
                        theta1 = np.sign(theta1) * m.radians(45)

                    targets = [theta1, theta2, theta3]

                    for i, joint_name in enumerate(JOINT_NAMES[leg]):
                        info = joint_info[joint_name]
                        pos_ref = targets[i]
                        vel_ref = 0.0

                        q = data.qpos[info['qpos_adr']]
                        qd = data.qvel[info['qvel_adr']]

                        vel_ref = (pos_ref - q) * control_hz * 0.1
                        desired_acc = gait.kp * (pos_ref - q) + gait.kd * (vel_ref - qd)
                        q_ddot_cmd[info['qvel_adr']] = desired_acc

                _m = np.zeros((model.nv, model.nv), dtype=np.float64)
                mujoco.mj_solveM(model, data, _m, np.eye(model.nv, dtype=np.float64))
                _m = np.linalg.inv(_m)

                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_fwdVelocity(model, data)
                bias_forces = data.qfrc_bias

                torque_id = (_m @ q_ddot_cmd) + bias_forces
                torque_id[6:] = (_m[6:, 6:] @ q_ddot_cmd[6:]) + bias_forces[6:]

                for leg in LEG_NAMES:
                    for joint in JOINT_NAMES[leg]:
                        info = joint_info[joint]
                        act_id = info['actuator_id']
                        dof_adr = info['qvel_adr']

                        c_torque = torque_id[dof_adr]
                        data.ctrl[act_id] = np.clip(c_torque, -100.0, 100.0)

            mujoco.mj_step(model, data)
            # if step_counter % 40 == 0:
            #     print(f"MujocoSim Step {step_counter}: base_z={data.qpos[2]:.3f}")

            for leg, joints in JOINT_NAMES.items():
                for joint in joints:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    body_id = model.jnt_bodyid[jid]
                    if body_id not in body_ids:
                        body_ids.append(body_id)

            # if step_counter < 30:
            #     bl_joints = JOINT_NAMES['BL']
            #     print(f"\n--- Frame {step_counter} (Time: {data.time:.4f}) ---")
            #     print(f"qpos: {[float(data.qpos[joint_info[j]['qpos_adr']]) for j in bl_joints]}")
            #     print(f"qvel: {[float(data.qvel[joint_info[j]['qvel_adr']]) for j in bl_joints]}")
            #     print(f"qacc_target: {[float(data.qacc[joint_info[j]['qvel_adr']]) for j in bl_joints]}")
            #     print(f"qfrc_inverse: {[float(data.qfrc_inverse[joint_info[j]['qvel_adr']]) for j in bl_joints]}")
            #     print(f"ctrl_clamped: {[float(data.ctrl[joint_info[j]['actuator_id']]) for j in bl_joints]}")

            viewer.sync()
            step_counter += 1

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    main()


        # def lerp(a, b, t):
        #     return a + t * (b - a)
        #
        # def logic_process(s, t, k):
        #     raw_msg = Float64MultiArray()
        #     raw_msg.data = [s, t, k] 
        #     self.raw_pub.publish(raw_msg)
        #
       #
        # def animate_transition(target_s, target_t, target_k, duration=1.0):
        #     start_time = time.time()
        #     initial_s = self.phi["tr_leg"]["shoulder"]
        #     initial_t = self.phi["tr_leg"]["thigh"]
        #     initial_k = self.phi["tr_leg"]["leg"]
        #
        #     def step():
        #         elapsed = time.time() - start_time
        #         fraction = min(elapsed / duration, 1.0)
        #
        #         current_s = lerp(initial_s, target_s, fraction)
        #         current_t = lerp(initial_t, target_t, fraction)
        #         current_k = lerp(initial_k, target_k, fraction)
        #
        #         for leg in self.phi.values():
        #             leg["shoulder"] = current_s
        #             leg["thigh"] = current_t 
        #             leg["leg"] = current_k 
        #
        #         if fraction < 1.0:
        #             root.after(20, step)
        #
        #     step()
