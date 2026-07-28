import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from typing import Callable
from envs.quadruped_env_walk import QuadrupedEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

URDF_PATH = "/home/ulone/ros2_ws/src/quins/urdf/quadruped.urdf"
PACKAGE_ROOT = "/home/ulone/ros2_ws/src/quins/"
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
N_ENVS = 12  # match to your CPU core count

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env():
    def _init():
        return QuadrupedEnv(
            urdf_path=URDF_PATH,
            package_root=PACKAGE_ROOT,
            render_mode="",
        )
    return _init


if __name__ == "__main__":
    # check_env needs one raw, unwrapped instance — not the vectorized/normalized one
    print("Checking environment")
    check_env(QuadrupedEnv(urdf_path=URDF_PATH, package_root=PACKAGE_ROOT, render_mode=""))
    print("Environment check passed")

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(0.0003),
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto"
    )

    print("STARTING 1 MILLION TIMESTEP TRAINING RUN")
    model.learn(total_timesteps=10000000, tb_log_name="PPO_walk_straight")
    model.save(f"{MODEL_DIR}/ppo_quadruped_straight_50")
    env.save(f"{MODEL_DIR}/vecnormalize50.pkl")

    env.close()
