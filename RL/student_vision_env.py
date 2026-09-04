import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from envs.quadruped_gym_env import QuadrupedResidualGymEnv

class VisionStudentEnv(gym.Env):
    """
    Phase 2 Distillation Wrapper[cite: 13, 14]: 
    Swaps privileged scandots for noisy depth camera images to train the Student policy.
    """
    def __init__(self):
        super().__init__()
        self.base_env = QuadrupedResidualGymEnv(render_mode='rgb_array')
        self.model = self.base_env.model
        self.data = self.base_env.data
        
        # Action space remains the same [v_x, v_y, omega_z]
        self.action_space = self.base_env.action_space
        
        # Proprioceptive states (35 dims) + 64x64 Depth Image
        self.observation_space = spaces.Dict({
            "proprioception": spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32),
            "depth_image": spaces.Box(low=0.0, high=255.0, shape=(1, 64, 64), dtype=np.float32)
        })
        
        self.renderer = mujoco.Renderer(self.model, height=64, width=64)
        self.renderer.enable_depth_rendering()

    def _get_student_obs(self, base_obs):
        # Strip off the 25 scandots from the base observation (last 25 elements)
        proprio = base_obs[:-25]
        
        # Render the depth camera array from the robot's front chassis
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, 
                               mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, self.renderer.scene)
        depth_array = self.renderer.render()
        
        # Normalize and clip depth map
        depth_array = np.clip(depth_array, 0.0, 5.0) 
        depth_array = (depth_array / 5.0) * 255.0
        depth_array = depth_array.astype(np.float32).reshape(1, 64, 64)
        
        return {"proprioception": proprio, "depth_image": depth_array}

    def reset(self, *, seed=None, options=None):
        base_obs, info = self.base_env.reset(seed=seed, options=options)
        return self._get_student_obs(base_obs), info

    def step(self, action):
        base_obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # The Student environment executes the same physics, but only "sees" depth
        student_obs = self._get_student_obs(base_obs)
        return student_obs, reward, terminated, truncated, info
