import mujoco
import mujoco.viewer
import threading
import numpy as np
import math as m
import time
import re
import tempfile
import xml.etree.ElementTree as ET
import tkinter as tk

BODY_WIDTH = 1.3334   
BODY_LENGTH = 2.2924  

class GaitParams:
    def __init__(self):
        self.gait_freq = 2.0
        self.x_off = 0.5
        self.z_off = 2.0
        self.step_len = 5.0
        self.step_h = 1.0

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

class KinematicsLogic:
    def __init__(self):
        self.l1 = 1.2925  
        self.l2 = 1.5005  
        self.l3 = 1.54138
        self.h_bw = BODY_WIDTH / 2.0
        self.h_bl = BODY_LENGTH / 2.0
        self.ee_io = np.zeros((4, 4, 4), dtype=float) 
        self.jvso = np.zeros((4, 3, 6)) 
        self.robot_world = np.array([0, 0, 0])

        def jvso_equation(s_ij):
            t = s_ij[0:3]
            r = s_ij[3:6]
            v = np.cross(-r, t)
            w = r 
            return np.concatenate((v, w))

        def screw_joint(j1, posx, posz, dir):
            match j1:
                case 1: return np.array([posx, 0, posz, dir, 0, 0]) 
                case 2: return np.array([posx, 0, posz, 0, 0, dir]) 
                case 3: return np.array([posx, -self.l2, posz, 0, 0, dir]) 
        
        for n in range(4):
            dir = 1 if n > 1 else -1
            hbl = self.h_bl if n == 0 or n == 3 else (-self.h_bl)
            self.ee_io[n] = np.array([
                [1, 0, 0,  hbl],
                [0, 1, 0, -self.l2 - self.l3],
                [0, 0, 1,  (self.h_bw + self.l1) * dir],
                [0, 0, 0,  1]
            ])
            for i in range(3):
                hbw = (self.h_bw + self.l1) * dir if i >= 1 else self.h_bw * dir
                bs_ni = screw_joint(i + 1, hbl, hbw, dir)
                self.jvso[n][i] = jvso_equation(bs_ni)

    def phase_to_index(self, leg_id):
        match leg_id:
            case 'FR': return 0
            case 'BR': return 1
            case 'BL': return 2
            case 'FL': return 3

    def get_init_pos(self, leg_id):
        n = self.phase_to_index(leg_id)
        return self.ee_io[n][0, 3], self.ee_io[n][1, 3], self.ee_io[n][2, 3]

    def ik(self, leg_id, x_r, y_r, z_r, knee_dir=-1):
        x = x_r
        y = y_r
        z = z_r
        x = x + self.h_bl if 'B' in leg_id else x - self.h_bl
        z = z - self.h_bw if 'L' in leg_id else z + self.h_bw

        x_a = x
        y_a = -(m.sqrt(y**2 + z**2 - self.l1**2))
        alpha_beta_denum = m.sqrt(y**2 + z**2)
        alpha_num = m.fabs(z)
        alpha_div = max(-1.0, min(1.0, alpha_num / alpha_beta_denum))
        alpha = m.acos(alpha_div)
        beta_num = self.l1
        beta_div = max(-1.0, min(1.0, beta_num / alpha_beta_denum))
        beta = m.acos(beta_div)
        phi_num = m.fabs(x_a)
        phi_denum = m.sqrt(x_a**2 + y_a**2)
        phi_div = max(-1.0, min(1.0, phi_num / phi_denum))
        phi = m.acos(phi_div)
        varphi_num = self.l2**2 + x_a**2 + y_a**2 - self.l3**2
        varphi_denum = 2 * self.l2 * m.sqrt(x_a**2 + y_a**2)
        varphi_div = max(-1.0, min(1.0, varphi_num / varphi_denum))
        varphi = m.acos(varphi_div)

        theta1 = alpha - beta if (('L' in leg_id and z > 0) or ('R' in leg_id and z < 0)) else m.pi - alpha - beta
        theta3_num = x_a**2 + y_a**2 - self.l2**2 - self.l3**2  
        theta3_denum = 2 * self.l2 * self.l3
        theta3_div = max(-1.0, min(1.0, theta3_num / theta3_denum)) 
        theta3 = m.acos(theta3_div) if knee_dir > 0 else -m.acos(theta3_div)
        theta2 = (m.pi / 2 if x_a > 0 else (-m.pi / 2)) + (-varphi if theta3 > 0 else varphi) + (-phi if x_a > 0 else phi)

        return theta1, theta2, theta3

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

# Prevent the dummy/base_link static chain from being fused into worldbody
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
            actuators_xml += f'    <position name="{joint}_servo" joint="{joint}" kp="300" kv="10"/>\n'
    actuators_xml += "</actuator>\n"

    mjcf_xml = mjcf_xml.replace('</worldbody>', f'</worldbody>\n{actuators_xml}')

    # Safely target the root body (no parent joint) to attach the freejoint
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

    k = KinematicsLogic()

    gait = GaitParams()
    threading.Thread(target=start_tuner_ui, args=(gait,), daemon=True).start()

    omega = 2.0 * m.pi * gait.gait_freq

    physics_hz = 1.0 / model.opt.timestep
    control_hz = 50.0
    decimation_steps = int(physics_hz / control_hz)
    step_counter = 0

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
                    targets = [theta1, theta2, theta3]

                    for i, joint_name in enumerate(JOINT_NAMES[leg]):
                        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_servo")
                        if actuator_id != -1:
                            data.ctrl[actuator_id] = targets[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            step_counter += 1

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == '__main__':
    main()
