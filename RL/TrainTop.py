import os
os.environ.setdefault("OMP_NUM_THREADS", "1")         # one BLAS thread per env process: many envs, small matrices
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

RL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RL_DIR.parent))
sys.path.insert(0, str(RL_DIR))
from RLTopEnv import RlTopEnv

# PPO settings from legged_gym (LeggedRobotCfgPPO): network, activation, clip, gamma, lambda, epochs,
# 4 minibatches, entropy, value loss weight, gradient clip, initial action noise.
# Two changes for a few CPU envs instead of 4096 GPU envs:
#   N_STEPS: legged_gym collects 24 steps x 4096 envs per update; with ~16 envs, 1024 steps keeps a usable batch
#   LEARNING_RATE: legged_gym's 1e-3 relies on its adaptive KL schedule, which SB3 does not have; 3e-4 is SB3's default
NET_ARCH = [512, 256, 128]
N_STEPS = 1024
LEARNING_RATE = 3.0e-4
PPO_KW = dict(
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    n_epochs=5,
    ent_coef=0.01,
    vf_coef=1.0,
    max_grad_norm=1.0,
)
INIT_NOISE_STD = 0.3
OUT_DIR = RL_DIR / "top_runs"


class RewardTermLogger(BaseCallback):
    """Logs each reward term's mean episode total and the fall rate to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.terms, self.falls, self.episodes = {}, 0, 0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done or "episode_reward_terms" not in info:
                continue
            self.episodes += 1
            self.falls += int(info["fell"])
            for k, v in info["episode_reward_terms"].items():
                self.terms.setdefault(k, []).append(v)
        return True

    def _on_rollout_end(self):
        for k, v in self.terms.items():
            self.logger.record(f"reward_terms/{k}", float(np.mean(v)))
        if self.episodes:
            self.logger.record("episodes/fall_rate", self.falls / self.episodes)
            self.logger.record("episodes/count", self.episodes)
        self.terms, self.falls, self.episodes = {}, 0, 0


def make_env(rank, seed):
    def _init():
        env = RlTopEnv(randomize=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="PPO training for the RL top layer (journal: WBC RL vs WBC MPC).")
    parser.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel env processes")
    parser.add_argument("--steps", type=int, default=10_000_000, help="total env steps (policy decisions)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu (recommended for MLP PPO) / cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)        # the 2 cores left free from --envs run the policy and the PPO update

    run_name = f"seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fns = [make_env(i, 1000 * args.seed) for i in range(args.envs)]
    venv = SubprocVecEnv(fns, start_method="forkserver") if args.envs > 1 else DummyVecEnv(fns) # type: ignore
    venv = VecMonitor(venv)
    # running mean / std of the observation (exported with the policy for the Jetson); reward is left as is
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0, gamma=PPO_KW["gamma"])

    rollout = N_STEPS * args.envs
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=N_STEPS,
        batch_size=rollout // 4,
        learning_rate=LEARNING_RATE,
        policy_kwargs=dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=torch.nn.ELU,
                           log_std_init=float(np.log(INIT_NOISE_STD))),
        tensorboard_log=str(OUT_DIR),
        seed=args.seed,
        device=args.device,
        verbose=1,
        **PPO_KW, # type: ignore
    )
    print(f"[train] {args.envs} envs, {rollout} steps per update, {args.steps} total, device {model.device}, "
          f"output {run_dir}", flush=True)

    save_every = max(rollout, 500_000 // args.envs * args.envs) // args.envs
    callbacks = [
        RewardTermLogger(),
        CheckpointCallback(save_freq=save_every, save_path=str(run_dir), name_prefix="ppo",
                           save_vecnormalize=True),
    ]
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name=run_name, progress_bar=False)
    finally:
        model.save(str(run_dir / "ppo_final"))
        venv.save(str(run_dir / "vecnormalize_final.pkl"))
        wall = time.perf_counter() - t0
        print(f"[train] {model.num_timesteps} steps in {wall / 60:.1f} min -> {model.num_timesteps / wall:.0f} steps/s",
              flush=True)
        print(f"[train] saved {run_dir / 'ppo_final.zip'} and {run_dir / 'vecnormalize_final.pkl'}", flush=True)
        venv.close()


if __name__ == "__main__":
    main()
