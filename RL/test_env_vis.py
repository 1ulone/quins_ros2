import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from envs.quadruped_env_stand import QuadrupedEnv

# UPDATE THESE PATHS to match your system
URDF_PATH = "/home/ulone/ros2_ws/src/quins/urdf/quadruped.urdf"
PACKAGE_ROOT = "/home/ulone/ros2_ws/src/quins/"

env = QuadrupedEnv(
    urdf_path=URDF_PATH,
    package_root=PACKAGE_ROOT,
    render_mode="human",
    target_range=3.0,
)

obs, info = env.reset()
print(f"Obs shape: {obs.shape}")  # Should be (48,)
print(f"Action space: {env.action_space}")
print(f"Obs space: {env.observation_space}")

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    time.sleep(0.02)
    
    if step % 40 == 0:
        print(f"Step {step}: reward={reward:.3f}, dist={info.get('distance', 0):.3f}, target={env.target_pos}, base_z={env.data.qpos[2]:.3f}")
    
    if terminated or truncated:
        print(f"Episode ended at step {step} (terminated={terminated}, truncated={truncated})")
        obs, info = env.reset()

env.close()
print("Environment test passed!")
