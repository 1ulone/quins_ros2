import sys
import os

from torch.fx.experimental.unification.unification_tools import first

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
        max_episode_steps: int = 10000,
        render_mode: str = "",
        ctrl_range: float = 100.0,
    ): 
        self.urdf_path = urdf_path
        self.package_root = package_root
        self.target_range = target_range
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.ctrl_range = ctrl_range
        self.phase = 0.0

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

        # obs_dim = 12 + 12 + 3 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.viewer = None
        self.step_count = 0
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.kinematics = KinematicsLogic()

        self.last_action = np.zeros(12, dtype=np.float32)
        self.previous_action = np.zeros(12, dtype=np.float32)

        self.feet_air_time = np.zeros(4)
        self.last_contact = np.zeros(4)

        foot_names = ['FL_foot', 'FR_foot', 'BL_foot', 'BR_foot']
        self._foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) 
            for name in foot_names
        ]

        self.default_joint_pos = np.copy(self.model.qpos0[7:])
        self.health_z_min = -2.3

        
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
                body.set('pos', '0 0 0.5')
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
        base_linear_velocity = self.data.qvel[:3]
        base_angular_velocity = self.data.qvel[3:6]
        
        dofs_position = self.data.qpos[7:].flatten()
        dofs_velocity = self.data.qvel[6:].flatten()
        
        obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        
        base_obs = np.concatenate([
            base_linear_velocity * obs_scale["linear_velocity"],
            base_angular_velocity * obs_scale["angular_velocity"],
            dofs_position * obs_scale["dofs_position"],
            dofs_velocity * obs_scale["dofs_velocity"],
            self.last_action
        ])
        
        # --- Restore Phase Encoding ---
        phase_offsets = {
            'FL': 0.0,
            'FR': np.pi,
            'BL': np.pi,
            'BR': 0.0,
        }
        
        phase_list = []
        for leg in ['FL', 'FR', 'BL', 'BR']:
            leg_phase = self.phase + phase_offsets[leg]
            phase_list.extend([np.sin(leg_phase), np.cos(leg_phase)])
            
        phase_obs = np.array(phase_list, dtype=np.float32)
        
        curr_obs = np.concatenate([base_obs, phase_obs]).clip(-100.0, 100.0).astype(np.float32)
        return curr_obs

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def get_reward(self):
        desired_velocity = np.array([0.5, 0.0])

        feet_contact_forces = np.linalg.norm(self.data.cfrc_ext[self._foot_body_ids], axis=1)
        curr_contact = feet_contact_forces > 1.0

        contact_filter = np.logical_or(curr_contact, self.last_contact)
        self.last_contact = curr_contact

        first_contact = (self.feet_air_time > 0.0) * contact_filter

        dt = self.model.opt.timestep * 10
        self.feet_air_time += dt

        feet_air_time_reward = np.sum((self.feet_air_time - 0.15) * first_contact) * 2.0
        self.feet_air_time *= ~contact_filter

        # --- D-Shape Trajectory Mechanics ---
        foot_velocities = self.data.cvel[self._foot_body_ids, 3:]
        rel_foot_x_vel = foot_velocities[:, 0] - self.data.qvel[0]
        swing_forward_reward = np.sum(np.maximum(0.0, rel_foot_x_vel) * (~curr_contact)) * 0.5
        foot_slip_cost = np.sum(np.square(foot_velocities[:, :2]) * curr_contact[:, np.newaxis]) * 1.0

        # --- Anti-Bounding Cost ---
        # Penalize front legs (0, 1) or back legs (2, 3) if they are both in the air or both on the ground
        front_bound = curr_contact[0] == curr_contact[1]
        back_bound = curr_contact[2] == curr_contact[3]
        bound_cost = (float(front_bound) + float(back_bound)) * 1.5

        # --- 2-Beat Trot Gait Enforcement ---
        phase_offsets = np.array([0.0, np.pi, np.pi, 0.0])
        leg_phases = self.phase + phase_offsets
        desired_contact = np.sin(leg_phases) <= 0
        phase_match = np.where(desired_contact == curr_contact, 1.0, -1.0)
        gait_reward = np.sum(phase_match) * 0.5

        # --- Positive Rewards ---
        vel_sqr_error = np.sum(np.square(desired_velocity - self.data.qvel[:2]))
        linear_vel_track_reward = np.exp(-vel_sqr_error / 0.25) * 2.0
        
        angular_vel_sqr_error = np.square(0.0 - self.data.qvel[5])
        angular_vel_track_reward = np.exp(-angular_vel_sqr_error / 0.25) * 1.0

        rewards = (
            linear_vel_track_reward + 
            angular_vel_track_reward + 
            feet_air_time_reward + 
            swing_forward_reward + 
            gait_reward
        )
        
        # --- Negative Costs ---
        torque_cost = np.sum(np.square(self.data.qfrc_actuator[-12:])) * 0.0002
        action_rate_cost = np.sum(np.square(self.last_action - self.previous_action)) * 0.01
        vertical_vel_cost = np.square(self.data.qvel[2]) * 2.0
        xy_angular_vel_cost = np.sum(np.square(self.data.qvel[3:5])) * 0.2
        joint_velocity_cost = np.sum(np.square(self.data.qvel[6:])) * 0.01
        joint_acceleration_cost = np.sum(np.square(self.data.qacc[6:])) * 2.5e-7
        default_joint_pos_cost = np.sum(np.square(self.data.qpos[7:] - self.default_joint_pos)) * 0.1

        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        gravity_vector = np.array(self.model.opt.gravity)
        projected_gravity = np.dot(gravity_vector, euler_orientation) * euler_orientation
        if np.linalg.norm(projected_gravity) != 0:
            projected_gravity = projected_gravity / np.linalg.norm(projected_gravity)
        orientation_cost = np.sum(np.square(projected_gravity[:2])) * 1.5

        costs = (
            torque_cost + 
            action_rate_cost + 
            vertical_vel_cost + 
            xy_angular_vel_cost + 
            joint_velocity_cost + 
            joint_acceleration_cost + 
            default_joint_pos_cost +
            orientation_cost +
            foot_slip_cost +
            bound_cost
        )

        reward = max(0.0, rewards - costs)

        return reward, self.data.qvel[0]

    def step(self, action):
        self.phase += 0.10
        self.previous_action = np.copy(self.last_action)
        self.last_action = np.copy(action)

        kp = 300.0
        kd = 25.0

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

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self.get_obs()
        reward, forward_vel = self.get_reward()

        terminated = bool(self.data.qpos[2] < self.health_z_min) 

        truncated = self.step_count >= self.max_episode_steps
        info = {"forward_velocity": forward_vel}
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.phase = 0.0

        mujoco.mj_resetData(self.model, self.data)
        
        self.target_pos = np.array([
            np.random.uniform(-self.target_range, self.target_range),
            np.random.uniform(-self.target_range, self.target_range),
            0.3
        ], dtype=np.float32)

        self.data.qpos[2] = 0.5
        self.feet_air_time = np.zeros(4)
        self.last_contact = np.zeros(4)

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
