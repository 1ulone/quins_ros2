import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import tempfile
import re
import xml.etree.ElementTree as ET
from quins.KinematicsLogic import KinematicsLogic

class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    JOINT_NAMES = [
        'tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint',   # FL
        'tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint',   # FR
        'bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint',   # BL
        'br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint',   # BR
    ]

    def __init__(
        self,
        urdf_path: str,
        package_root: str,
        target_range: float = 5.0,
        max_episode_steps: int = 1000,
        render_mode: str = "",
        ctrl_range: float = 100.0,
    ): 
        self.urdf_path = urdf_path
        self.package_root = package_root
        self.target_range = target_range
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.ctrl_range = ctrl_range

        self.model, self.data = self.build_mjcf()

        self.joint_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.joint(name).id] for name in self.JOINT_NAMES
        ]) 
        self.joint_qvel_adr = np.array([
            self.model.jnt_dofadr[self.model.joint(name).id] for name in self.JOINT_NAMES
        ])
        self.actuator_ids = np.array([
            self.model.actuator(f"{name}_motor").id for name in self.JOINT_NAMES
        ])

        obs_dim = 12 + 12 + 3 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.viewer = None
        self.step_count = 0
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.kinematics = KinematicsLogic()
        
    def build_mjcf(self):
        with open(self.urdf_path, 'r') as file:
            urdf_xml = file.read()

        urdf_xml = urdf_xml.replace('package://quins/', self.package_root)
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
        for joint in self.JOINT_NAMES:
            actuators_xml += f'    <motor name="{joint}_motor" joint="{joint}" gear="1" ctrllimited="true" ctrlrange="-100 100"/>\n'
        actuators_xml += "</actuator>\n"

        mjcf_xml = mjcf_xml.replace('</worldbody>', f'</worldbody>\n{actuators_xml}')

        root_xml = ET.fromstring(mjcf_xml)
        worldbody = root_xml.find('worldbody')
        assert worldbody is not None

        for body in worldbody.findall('body'):
            if body.find('joint') is None:
                body.set('pos', '0 0 2.15')
                ET.SubElement(body, 'freejoint', name='root_floating_base')
                break

        mjcf_xml = ET.tostring(root_xml, encoding='unicode')

        model = mujoco.MjModel.from_xml_string(mjcf_xml)
        
        # Collision filters matching MujocoSim.py
        model.geom_contype[:] = 1
        model.geom_conaffinity[:] = 0
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id != -1:
            model.geom_contype[floor_id] = 0
            model.geom_conaffinity[floor_id] = 1

        data = mujoco.MjData(model)

        return model, data

    def get_obs(self):
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_qvel_adr]
        base_pos = self.data.qpos[0:3]
        base_quat = self.data.qpos[3:7]
        base_vel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        return np.concatenate([
            joint_pos, joint_vel, base_pos, base_quat, base_vel, base_angvel, self.target_pos
        ]).astype(np.float32)

    def get_reward(self):
        base_xy = self.data.qpos[0:2]
        dist = np.linalg.norm(base_xy - self.target_pos[:2])
        upright = self.data.qpos[3] ** 2
        to_target = self.target_pos[:2] - base_xy
        if dist > 1e-6:
            vel_toward = np.dot(self.data.qvel[0:2], to_target / dist)
        else:
            vel_toward = 0.0
        action_penalty = np.sum(np.square(self.last_action))
        reward = -5.0 * dist + 0.5 * upright + 0.1 * vel_toward - 0.001 * action_penalty
        return reward, dist

    def step(self, action):
        self.last_action = action
        kp = 100.0
        kd = 20.0

        # Set floating base acceleration targets to zero
        self.data.qacc[0:6] = 0.0

        cartesian_targets = action.reshape(4, 3) * 0.15
        actual_z = self.data.qpos[2]
        kz = 0.6
        x_off = 0.20
        z_off = 1.25
        z_err = kz * (z_off - actual_z)
        
        target_qpos = []
        for i, leg in enumerate(['FL', 'FR', 'BL', 'BR']):
            ix, iy, iz = self.kinematics.get_init_pos(leg)
            
            # 0.30 x_off + 2.15 ride height
            tx = ix + x_off + cartesian_targets[i][0]
            ty = z_off + z_err + cartesian_targets[i][1]
            tz = iz + cartesian_targets[i][2]
            
            theta1, theta2, theta3 = self.kinematics.ik(leg, tx, ty, tz)
            target_qpos.extend([theta1, theta2, theta3])

        for i, actuator_id in enumerate(self.actuator_ids):
            qpos_adr = self.joint_qpos_adr[i]
            qvel_adr = self.joint_qvel_adr[i]

            q = self.data.qpos[qpos_adr]
            qd = self.data.qvel[qvel_adr]

            pos_ref = target_qpos[i]
            vel_ref = 0.0

            desired_acc = kp * (pos_ref - q) + kd * (vel_ref - qd)
            self.data.qacc[qvel_adr] = desired_acc

        mujoco.mj_inverse(self.model, self.data)

        for i, actuator_id in enumerate(self.actuator_ids):
            dof_adr = self.joint_qvel_adr[i]
            computed_torque = self.data.qfrc_inverse[dof_adr]
            self.data.ctrl[actuator_id] = np.clip(computed_torque, -self.ctrl_range, self.ctrl_range)

        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self.get_obs()
        reward, dist = self.get_reward()
        
        terminated = bool(self.data.qpos[2] < -2.35) 
        truncated = self.step_count >= self.max_episode_steps
        info = {"distance": dist}
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        
        self.target_pos = np.array([
            np.random.uniform(-self.target_range, self.target_range),
            np.random.uniform(-self.target_range, self.target_range),
            0.3
        ], dtype=np.float32)

        self.data.qpos[2] = 2.15

        leg_map = {'FL': 'tl', 'FR': 'tr', 'BL': 'bl', 'BR': 'br'}
        for leg in ['FL', 'FR', 'BL', 'BR']:
            ix, iy, iz = self.kinematics.get_init_pos(leg)
            
            # Initial target generation aligned perfectly with step() baseline
            theta1, theta2, theta3 = self.kinematics.ik(leg, ix + 0.30, 2.15, iz)
            
            prefix = leg_map[leg]
            for i, joint_name in enumerate([f'{prefix}_shoulder_joint', f'{prefix}_thigh_joint', f'{prefix}_leg_joint']):
                if joint_name in self.JOINT_NAMES:
                    idx = self.JOINT_NAMES.index(joint_name)
                    self.data.qpos[self.joint_qpos_adr[idx]] = [theta1, theta2, theta3][i]

        mujoco.mj_forward(self.model, self.data)
        return self.get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.Renderer(self.model)
            self.viewer.update_scene(self.data)
            return self.viewer.render()

    def close(self):
        if self.viewer is not None:
            import time
            time.sleep(0.1)
            self.viewer.close()
            self.viewer = None
