import os
import sys
import time
import imageio
from numpy import heaviside
import torch
import mujoco.viewer
from pathlib import Path

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from RL.student_vision_env import VisionStudentEnv
from RL.distill_policy import VisionStudentPolicy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(current_dir, 'logs', 'student_vision_policy.pth')

    if not os.path.exists(model_path):
        print(f"Error: Trained student policy not found at {model_path}. Run distill_policy.py first.")
        sys.exit(1)

    # 1. Instantiate the Student Environment
    env = VisionStudentEnv()
    base_env = env.base_env

    # 2. Load the trained PyTorch Vision Policy
    policy = VisionStudentPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    obs, _ = env.reset()

    print("Running Phase 3: Autonomous Vision Execution...")

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    video_path = str(current_dir / 'rl_3p_model.mp4')
    video_writer = imageio.get_writer(video_path, fps=60)

    # 3. Run the Execution Loop with MuJoCo Viewer
    try:
        with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # Convert dictionary observations to tensors
                proprio = torch.tensor(obs["proprioception"], dtype=torch.float32).unsqueeze(0).to(device)
                depth = torch.tensor(obs["depth_image"], dtype=torch.float32).unsqueeze(0).to(device)

                # Predict target velocity and heading using the CNN
                with torch.no_grad():
                    action = policy(proprio, depth).cpu().numpy().flatten()

                # Step the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                
                viewer.sync()

                mujoco.mjv_updateScene(env.model, env.data, viewer.opt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
                video_writer.append_data(renderer.render())
                
                if terminated or truncated:
                    obs, _ = env.reset()

                elapsed = time.time() - step_start
                if elapsed < (1.0 / base_env.control_hz):
                    time.sleep((1.0 / base_env.control_hz) - elapsed)
    finally:
        video_writer.close()
        env.close()

if __name__ == '__main__':
    main()
