import os
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import imageio
from stable_baselines3 import PPO

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from envs.quadruped_gym_env import QuadrupedResidualGymEnv

def main():
    scene_path = str(project_root / 'urdf' / 'scene.xml')
    urdf_path = str(project_root / 'urdf' / 'quadruped.urdf')
    model_path = str(current_dir / 'logs' / 'residual_ppo_final.zip')

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Did the training finish?")
        sys.exit(1)

    # 1. Instantiate the Environment
    env = QuadrupedResidualGymEnv(
        scene_path=scene_path,
        urdf_path=urdf_path,
        render_mode='human'
    )

    # 2. Load the trained policy
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset()

    # 3. Setup the Recorder
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    video_path = str(current_dir / 'rl_evaluation.mp4')
    video_writer = imageio.get_writer(video_path, fps=env.control_hz)

    print(f"Recording simulation to {video_path} at {env.control_hz} FPS...")

    # 4. Run the MuJoCo Viewer Loop
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # Predict action using the trained PyTorch policy
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)

                viewer.sync()
                
                # Update scene and capture the frame for the video
                mujoco.mjv_updateScene(env.model, env.data, viewer.opt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
                video_writer.append_data(renderer.render())
                
                if terminated or truncated:
                    obs, _ = env.reset()

                elapsed = time.time() - step_start
                if elapsed < (1.0 / env.control_hz):
                    time.sleep((1.0 / env.control_hz) - elapsed)
    finally:
        # Crucial: Close the writer when you exit the viewer to prevent a corrupted mp4
        video_writer.close()
        print("Recording saved successfully.")

if __name__ == '__main__':
    main()
