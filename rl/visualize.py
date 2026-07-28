import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from envs.quadruped_env_walk import QuadrupedEnv
from stable_baselines3 import PPO

URDF_PATH = "/home/ulone/ros2_ws/src/quins/urdf/quadruped.urdf"
PACKAGE_ROOT = "/home/ulone/ros2_ws/src/quins/"
MODEL_PATH = "./models/ppo_quadruped_straight"

env = QuadrupedEnv(
    urdf_path=URDF_PATH,
    package_root=PACKAGE_ROOT,
    render_mode="human"
)

model = PPO.load(MODEL_PATH, env=env)
obs, info = env.reset()

for step in range(2000):
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
