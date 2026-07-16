import mujoco
import mujoco.viewer
import threading
import math as m
import numpy as np
import time
import re
import tempfile
import xml.etree.ElementTree as ET
import tkinter as tk
import scipy.linalg
from KinematicsLogic import KinematicsLogic

BODY_WIDTH = 1.3334   
BODY_LENGTH = 2.2924  

class GaitParams:
    def __init__(self):
        self.gait_freq = 2.40
        self.x_off = 0.45
        self.z_off = 1.80
        self.step_len = 2.80
        self.step_h = 1.50

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

    root.mainloop()

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
    LEG_NAMES = ['FL', 'FR', 'BL', 'BR']

    phase_offsets = {
        'FL': 0.0,
        'BR': m.pi / 2.0,
        'FR': m.pi,
        'BL': 3.0 * m.pi / 2.0,
    }

    JOINT_NAMES = {
        'FL': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
        'FR': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
        'BL': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
        'BR': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint'],
    }

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
            body.set('pos', '0 0 3')
            ET.SubElement(body, 'freejoint', name='root_floating_base')
            break

    mjcf_xml = ET.tostring(root_xml, encoding='unicode')

    model = mujoco.MjModel.from_xml_string(mjcf_xml)
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

    Kp = 80.0
    Kd = 4.0

    k = KinematicsLogic()

    gait = GaitParams()
    threading.Thread(target=start_tuner_ui, args=(gait,), daemon=True).start()

    physics_hz = 1.0 / model.opt.timestep
    control_hz = 50.0
    decimation_steps = int(physics_hz / control_hz)
    step_counter = 0
    body_ids = [1]

    dt_control = 1.0 / control_hz
    ref_state = {
        joint: { 'pos': None, 'vel': 0.0, 'acc': 0.0 }
        for joints in JOINT_NAMES.values()
        for joint in joints
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if step_counter % decimation_steps == 0:
                t = data.time
                omega = 2.0 * m.pi * gait.gait_freq
                phase_now = (omega * t) % (2.0 * m.pi)

                for leg in LEG_NAMES:
                    leg_phase = (phase_now + phase_offsets[leg]) % (2.0 * m.pi)
                    xl, yl, zl = trajectory_controller(leg_phase, gait.step_len, gait.step_h)
                    ix, iy, iz = k.get_init_pos(leg)

                    tx = ix + xl + gait.x_off
                    ty = gait.z_off - yl
                    tz = iz + zl
                    theta1, theta2, theta3 = k.ik(leg, tx, ty, tz)
                    # targets = [theta1, theta2, theta3]

                    # for i, joint_name in enumerate(JOINT_NAMES[leg]):
                    #     info = joint_info[joint_name]
                    #     ref = ref_state[joint_name]
                    #
                    #     pos_ref = targets[i]
                    #     prev_pos = ref['pos']
                    #     vel_ref = 0.0 if prev_pos is None else (pos_ref - prev_pos) / dt_control
                    #     acc_ref = 0.0 if prev_pos is None else (vel_ref - ref['vel']) / dt_control
                    #
                    #     ref['pos'] = pos_ref
                    #     ref['vel'] = vel_ref
                    #     ref['acc'] = acc_ref
                    #
                    #     q = data.qpos[info['qpos_adr']]
                    #     qd = data.qvel[info['qvel_adr']]
                    #

                spatial_inertias = [k.get_spatial_inertia(model, data, bid) for bid in body_ids]
                I_global = k.block_diag(*spatial_inertias)

                support_legs = []
                for leg in LEG_NAMES:
                    leg_link = f"{leg.lower()[0]}_leg"
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg_link)
                    if data.xpos[body_id][2] < (gait.z_off * 0.1):
                        support_legs.append(leg_link)

                C, C_T = get_jacobian_constraint(model, data, support_legs)

                mujoco.mj_rne(model, data, 0, data.qvel)
                f_p = -data.qfrc_bias

                C_a = np.zeros(C_T.shape[0])
                row_idx = 0
                for body_name in support_legs:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    jacp_dot = np.zeros((3, model.nv))
                    mujoco.mj_jacDot(model, data, jacp_dot, None, data.xpos[body_id], body_id)
                    C_a[row_idx:row_idx+3] = -jacp_dot @ data.qvel
                    row_idx += 3

                nv = model.nv
                nc = C.shape[1]

                top_row = np.hstack((-I_global, C))
                bottom_row = np.hstack((C_T, np.zeros((nc, nc))))

                A = np.vstack((top_row, bottom_row))
                b = np.concatenate((f_p, C_a))

                try:
                    x = scipy.linalg.solve(A, b)
                    accelerations = x[:nv]
                    contact_forces = x[nv:]

                    data.qacc[:] = accelerations
                    mujoco.mj_inverse(model, data)

                    for leg in LEG_NAMES:
                        for joint in JOINT_NAMES[leg]:
                            info = joint_info[joint]
                            act_id = info['actuator_id']
                            dof_adr = info['qvel_adr']

                            computed_torque = data.qfrc_inverse[dof_adr]
                            data.ctrl[act_id] = np.clip(computed_torque, -100, 100)
                except scipy.linalg.LinAlgError:
                    data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)
            for leg, joints in JOINT_NAMES.items():
                for joint in joints:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    body_id = model.jnt_bodyid[jid]
                    if body_id not in body_ids:
                        body_ids.append(body_id)


            viewer.sync()
            step_counter += 1

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    main()
