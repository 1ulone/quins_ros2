import sys
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

RL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RL_DIR.parent))
sys.path.insert(0, str(RL_DIR))
from RLTopEnv import RlTopEnv, TEST_SCRIPT
from LOGIC.RLTopLogic import N_ACTION


def load_policy(run_dir, model_name, vecnorm_name):
    """Trained policy as a plain function: raw obs -> action (deterministic), with the training normalization."""
    model = PPO.load(str(run_dir / model_name), device="cpu")
    norm = VecNormalize.load(str(run_dir / vecnorm_name), DummyVecEnv([lambda: RlTopEnv(randomize=False)]))
    norm.training = False
    def policy(obs):
        action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
        return action
    return policy


def run_episode(env, act, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    terminated = None
    while not done:
        obs, r, terminated, truncated, _ = env.step(act(obs))
        total += r
        done = terminated or truncated
    return total, terminated, dict(env.reward_sums)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL top layer against the zero action (WBC + PD).")
    parser.add_argument("run_dir", help="folder in RL/top_runs with the model and its vecnormalize file")
    parser.add_argument("--model", default="ppo_final.zip")
    parser.add_argument("--vecnorm", default="vecnormalize_final.pkl")
    parser.add_argument("--episodes", type=int, default=20, help="randomized episodes per controller")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    policy = load_policy(run_dir, args.model, args.vecnorm)
    zero = lambda obs: np.zeros(N_ACTION)

    # 1) the journal script on the nominal plant, same summary as the zero-action test
    for name, act in (("ZERO ACTION (WBC + PD)", zero), ("TRAINED POLICY", policy)):
        env = RlTopEnv(randomize=False, script=TEST_SCRIPT, episode_s=25.0)
        total, _, _ = run_episode(env, act, seed=0)
        print(f"\n########## {name}: journal script, nominal plant, return {total:.2f} ##########")
        env.summary()

    # 2) randomized plants (mass, friction, motor, pushes, random state sequence), same seeds for both
    print(f"\n########## RANDOMIZED: {args.episodes} episodes each, same seeds ##########")
    for name, act in (("zero action", zero), ("trained policy", policy)):
        env = RlTopEnv(randomize=True)
        returns, falls, state_cost = [], 0, []
        for ep in range(args.episodes):
            total, fell, sums = run_episode(env, act, seed=10_000 + ep)
            returns.append(total)
            falls += int(fell) # type: ignore
            state_cost.append(sums["mpc_state_cost"])
        print(f"{name:15s}: return {np.mean(returns):6.2f} +- {np.std(returns):5.2f}   falls {falls}/{args.episodes}   "
              f"MPC state cost {np.mean(state_cost):7.2f}", flush=True)


if __name__ == "__main__":
    main()
