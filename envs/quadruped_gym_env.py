import os
import math as m
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.FOSMCLogic import FOSMC
from LOGIC.DynamicsLogic import QuadrupedDynamics
from LOGIC.MpcLogic import CentroidalMPC

class QuadrupedResidualGymEnv(gym.Env):
    """
    Gymnasium environment wrapping MuJoCo physics with Centroidal MPC and FOSMC baselines[cite: 7, 11].
    The RL agent outputs residual joint torque corrections (Delta tau)[cite: 10, 11].
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, scene_path=None, urdf_path=None, render_mode=None):
        super().__init__()
        
        # Resolve paths
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scene_path = scene_path or os.path.join(pkg_root, 'urdf', 'scene.xml')
        self.urdf_path = urdf_path or os.path.join(pkg_root, 'urdf', 'quadruped.urdf')
        self.total_steps = 0
        self.current_scandots = np.zeros(25, dtype=np.float32)

        # Load MuJoCo Model and Data[cite: 7]
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)

        # Cache Joint and Actuator Info[cite: 7]
        self.joint_info = {}
        self.adr_map = []
        for leg in LEG_NAMES:
            for joint in JOINT_NAMES[leg]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                info = {
                    'qpos_adr': self.model.jnt_qposadr[jid],
                    'qvel_adr': self.model.jnt_dofadr[jid],
                    'actuator_id': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_motor"),
                }
                self.joint_info[joint] = info
                self.adr_map.append(info['qvel_adr'])

        self.foot_body_names = ['tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link']
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.foot_body_names]

        # Controllers Setup[cite: 7, 11]
        self.dyn = QuadrupedDynamics(self.urdf_path)
        self.mpc = CentroidalMPC(self.urdf_path, n=10, dt=0.02)
        self.fosmc = FOSMC(
            dof=12,
            dt=self.model.opt.timestep,
            lam=0.5,
            alpha=1.5,
            Ke1=24.0,
            Ke2=5.0,
            Ks=25.0,
            Kr=10.0,
            gamma_c=0.01,
            gamma_a=0.01
        )

        # Timing and Rates[cite: 7]
        self.physics_hz = int(1.0 / self.model.opt.timestep)
        self.control_hz = 50  # RL / High-level step frequency
        self.decimation = int(self.physics_hz / self.control_hz)
        self.mpc_decimation = int(self.physics_hz / 50)
        self.optimal_foot_forces = np.zeros(12)

        # Gait trajectory command cache
        self.cmd = {
            "q_des": np.zeros(12),
            "qd_des": np.zeros(12),
            "qdd_des": np.zeros(12),
            "foot_forces": np.zeros((4, 3)),
            "is_stance": [True, True, True, True]
        }

        self.logic = GaitLogic({
            "walk_points": self._cb_walk_points,
            "jump_points": lambda *args: None,
            "transition_cb": lambda *args: None,
            "raw_tune_cb": lambda *args: None,
            "graph": lambda *args: None
        })

        # Observation Space: Base RPY (3), LinVel (3), AngVel (3), Joint Pos (12), Joint Vel (12), Rel Target (2), Scandots (25)
        # Total observation dimension = 60
        obs_dim = 3 + 3 + 3 + 12 + 12 + 2 + 25
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action Space: High-Level Navigation [v_x, v_y, omega_z]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.target_pos = np.array([10.0, 0.0]) # Navigation Goal X, Y past the obstacles
        self.step_counter = 0
        self.render_mode = render_mode

    def _cb_walk_points(self, points_data):
        if not points_data: return
        pt = points_data[0]
        self.cmd["q_des"] = np.array(pt["positions"])
        self.cmd["qd_des"] = np.array(pt["velocities"])
        self.cmd["qdd_des"] = np.array(pt["accelerations"])
        self.cmd["is_stance"] = pt["is_stance"]
        self.cmd["foot_forces"] = pt["foot_forces"]

    def _get_obs(self):
        qw, qx, qy, qz = self.data.qpos[3:7]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = m.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = m.asin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = m.atan2(siny_cosp, cosy_cosp)
        rpy = np.array([roll, pitch, yaw], dtype=np.float32)

        lin_vel = np.array(self.data.qvel[0:3], dtype=np.float32)
        ang_vel = np.array(self.data.qvel[3:6], dtype=np.float32)
        q_act, qd_act = self._get_joint_state()

        base_x, base_y, base_z = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]
        rel_target = np.array([self.target_pos[0] - base_x, self.target_pos[1] - base_y], dtype=np.float32)

        # Generate Scandots (Simulated LIDAR)[cite: 21]
        # Casts a 5x5 grid of rays straight down around the robot to measure terrain depth
        scandots = np.zeros(25, dtype=np.float32)
        grid_offsets = np.linspace(-0.5, 0.5, 5)
        
        geomid = np.array([-1], dtype=np.int32)
        idx = 0
        for dx in grid_offsets:
            for dy in grid_offsets:
                # Transform grid offsets by current yaw to keep LIDAR local to robot heading
                local_dx = dx * m.cos(yaw) - dy * m.sin(yaw)
                local_dy = dx * m.sin(yaw) + dy * m.cos(yaw)
                
                ray_start = np.array([base_x + local_dx, base_y + local_dy, base_z])
                ray_dir = np.array([0.0, 0.0, -1.0])
                
                dist = mujoco.mj_ray(self.model, self.data, ray_start, ray_dir, None, 1, -1, geomid)
                # If ray hits nothing, cap at 2.0 meters
                scandots[idx] = dist if dist > 0 else 2.0
                idx += 1

        self.current_scandots = scandots

        return np.concatenate([rpy, lin_vel, ang_vel, q_act, qd_act, rel_target, scandots])

    def _get_joint_state(self):
        q_act = np.zeros(12)
        qd_act = np.zeros(12)
        idx = 0
        for leg in LEG_NAMES:
            for j in JOINT_NAMES[leg]:
                info = self.joint_info[j]
                q_act[idx] = self.data.qpos[info['qpos_adr']]
                qd_act[idx] = self.data.qvel[info['qvel_adr']]
                idx += 1
        return q_act, qd_act

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        difficulty = np.clip(self.total_steps / 1_000_000, 0.0, 1.0)
        
        # 0. Procedural Terrain Curriculum[cite: 14]
        # Randomize obstacle geometry in the MuJoCo XML graph before physics start
        try:
            hurdle_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle_1")
            hurdle_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle_2")
            # stair_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "stair_1")
            # stair_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "stair_2")
            
            # Randomize Heights and Placements
            self.model.geom_pos[hurdle_1_id][0] = np.random.uniform(-10.0, 10.0)    # Hurdle 1 X pos
            self.model.geom_pos[hurdle_1_id][1] = np.random.uniform(-10.0, 10.0)    # Hurdle 1 Y pos
            self.model.geom_size[hurdle_1_id][2] = np.random.uniform(0.05, 0.15) * difficulty # Hurdle 1 Height
            
            self.model.geom_pos[hurdle_2_id][0] = np.random.uniform(-10.0, 10.0)    # Hurdle 2 X pos
            self.model.geom_pos[hurdle_2_id][1] = np.random.uniform(-10.0, 10.0)    # Hurdle 2 Y pos
            self.model.geom_size[hurdle_2_id][2] = np.random.uniform(0.1, 0.25) * difficulty  # Hurdle 2 Height
            
            # self.model.geom_size[stair_1_id][2] = np.random.uniform(0.05, 0.2) * difficulty  # Stair 1 Height
            # self.model.geom_size[stair_2_id][2] = np.random.uniform(0.15, 0.35)  # Stair 2 Height
            
            mujoco.mj_forward(self.model, self.data)
        except Exception:
            pass # Failsafe if running flat-ground fallback XML
            
        # Set nominal spawn height
        self.data.qpos[2] = 0.0
        self.data.qpos[3] = 1.0  # Quaternion w

        # Reset gait generator & low-level controller states
        self.step_counter = 0
        self.fosmc.frac_deriv.buffer.clear()
        self.optimal_foot_forces = np.zeros(12)
        
        # 1. Command the walk state
        self.logic.update_state("WALK")
        
        # 2. Pre-calculate the first frame of the IK trajectory
        self.logic.current_q, self.logic.current_q_dot = self._get_joint_state()
        self.logic.loop_step()
        
        # 3. Snap the physical MuJoCo joints to the bent IK targets before physics start
        self.data.qpos[7:19] = self.cmd["q_des"]
        mujoco.mj_forward(self.model, self.data)
        
        # Randomize a navigation waypoint between 3 to 6 meters away (The "Ifs" Condition)
        angle = np.random.uniform(-m.pi/4, m.pi/4)
        distance = np.random.uniform(3.0, 6.0)
        self.target_pos = np.array([distance * m.cos(angle), distance * m.sin(angle)])
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):

        try:
            terrain_z = self.data.qpos[2] - np.mean(self.current_scandots)
        except AttributeError:
            terrain_z = 0.0

        target_z = terrain_z + 0.25
        # Decode high-level navigation action from the AI: [v_x, v_y, omega_z]
        # v_x: [-1.0, 2.0] m/s (allow slight reverse, mostly forward)
        # v_y: [-0.5, 0.5] m/s (strafe)
        # omega_z: [-1.0, 1.0] rad/s (turn)
        target_vx = np.clip(action[0] * 1.5 + 0.5, -1.0, 2.0) 
        target_vy = np.clip(action[1] * 0.5, -0.5, 0.5)
        target_wz = np.clip(action[2] * 1.0, -1.0, 1.0)

        # Run decimation sub-steps across the physics loop
        for _ in range(self.decimation):
            # 1. Update high-level gait targets
            if self.step_counter % int(self.physics_hz / self.logic.control_rate) == 0:
                # Synchronize AI's commanded velocity into GaitLogic (The "Result")
                self.logic.step_len = target_vx
                self.logic.target_yaw += target_wz * (1.0 / self.logic.control_rate)
                
                q_act, qd_act = self._get_joint_state()
                self.logic.current_q = q_act
                self.logic.current_q_dot = qd_act
                self.logic.loop_step()

            # 2. Update Centroidal MPC forces
            if self.step_counter % self.mpc_decimation == 0:
                p_act = self.data.qpos[0:3]
                v_act = self.data.qvel[0:3]
                w_act = self.data.qvel[3:6]

                qw, qx, qy, qz = self.data.qpos[3:7]
                sinp = 2.0 * (qw * qy - qz * qx)
                pitch = m.asin(np.clip(sinp, -1.0, 1.0))
                rpy_act = np.array([self.logic.current_roll, pitch, self.logic.current_yaw])

                foot_positions = [self.data.xpos[fid] for fid in self.foot_ids]
                planned_schedule = [self.cmd["is_stance"] for _ in range(self.mpc.n)]

                opti, U_f, p_ref, v_ref = self.mpc.setup_problem(
                    p_act, v_act, rpy_act, w_act, foot_positions, planned_schedule
                )
                
                # Pass AI's target velocity directly into the MPC solver (The "Execution")
                opti.set_value(p_ref, np.array([0, 0, target_z])) # Z-hover remains 0.0 based on floor physics
                opti.set_value(v_ref, np.array([target_vx, target_vy, 0.0]))

                try:
                    sol = opti.solve()
                    self.optimal_foot_forces = np.array(sol.value(U_f[:, 0])).flatten()
                except Exception:
                    pass

            # 3. Compute Baseline FOSMC Feedback[cite: 7, 8]
            q_act, qd_act = self._get_joint_state()
            pd_torques_arr, s_current = self.fosmc.compute(
                q=q_act,
                q_dot=qd_act,
                q_d=self.cmd["q_des"],
                q_dot_d=self.cmd["qd_des"]
            )
            pd_torques_arr = np.nan_to_num(pd_torques_arr, nan=0.0, posinf=1500.0, neginf=-1500.0)

            # 4. Map 12-DoF signals to 18-DoF MuJoCo model
            pd_torques = np.zeros(self.model.nv)
            for i, adr in enumerate(self.adr_map):
                pd_torques[adr] = float(pd_torques_arr[i])
                # We removed the low-level residual torque since AI is acting as a high-level planner now

            # 5. Compute MPC Feedforward Torques via Foot Jacobians[cite: 7, 12]
            tau_ff = np.zeros(self.model.nv)
            jacobians = self.dyn.get_foot_jacobians(self.data.qpos)
            for i in range(4):
                if self.cmd["is_stance"][i]:
                    F_i = self.optimal_foot_forces[3*i : 3*i+3]
                    # F_i = self.cmd["foot_forces"][i]
                    tau_ff += jacobians[i].T @ F_i

            # 6. Apply Superposed Torques
            for leg in LEG_NAMES:
                for j in JOINT_NAMES[leg]:
                    info = self.joint_info[j]
                    adr = info['qvel_adr']
                    # Apply MPC Strategic Force + FOSMC Baseline Tracking
                    final_torque = tau_ff[adr] + pd_torques[adr]
                    self.data.ctrl[info['actuator_id']] = np.clip(final_torque, -1500.0, 1500.0)

            mujoco.mj_step(self.model, self.data)
            self.step_counter += 1

        obs = self._get_obs()
        
        # Navigation Reward Formulation
        base_x, base_y, base_pos_z = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]
        
        # Distance to the randomly spawned waypoint
        dist_to_target = m.hypot(self.target_pos[0] - base_x, self.target_pos[1] - base_y)
        
        # Reward approaching the target, penalize deviating from height
        progress_reward = -dist_to_target
        survival_reward = 1.0
        posture_penalty = 5.0 * abs(base_pos_z - target_z) 

        reward = progress_reward + survival_reward - posture_penalty

        # Termination Conditions
        terminated = False
        if base_pos_z < -0.2:
            terminated = True
            reward -= 50.0

        # Success condition (reached the waypoint)
        if dist_to_target < 0.5:
            terminated = True
            reward += 100.0

        truncated = self.step_counter >= 10000  # Max episode steps
        self.total_steps += 1

        return obs, reward, terminated, truncated, {}
