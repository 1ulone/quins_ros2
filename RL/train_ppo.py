import os
import sys
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from envs.quadruped_gym_env import QuadrupedResidualGymEnv

def train():
    # Model & XML paths
    scene_path = str(project_root / 'urdf' / 'scene.xml')
    urdf_path = str(project_root / 'urdf' / 'quadruped.urdf')
    log_dir = str(current_dir / 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 1. Instantiate the Environment
    env = QuadrupedResidualGymEnv(
        scene_path=scene_path,
        urdf_path=urdf_path,
        render_mode=None
    )

    # 2. Checkpoint Callback (Saves model every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix="residual_ppo_fosmc"
    )

    # 3. Setup PPO Agent
    device = "cpu"
    print(f"Training PPO on device: {device}")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
        device=device
    )

    # 4. Train the Residual Policy
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 5. Save Final Policy
    final_model_path = os.path.join(log_dir, "residual_ppo_final.zip")
    model.save(final_model_path)
    print(f"Training Complete. Policy saved to: {final_model_path}")

if __name__ == '__main__':
    train()
