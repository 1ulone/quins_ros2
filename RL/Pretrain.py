import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from LOGIC.FOSMCLogic import FOSMC
from MUJOCO.MujocoSim import main as run_episode
from RL.MainTuner import RLTuner

N_EPISODES = 20
EP_T = 32.0                 # 2 s standing + 30 s walking (AUTO_STATE in MujocoSim starts WALK at 2 s)
SAVE_PATH = os.path.join(current_dir, "tuner_pretrained.npz")
N_TEST = 5          # fixed randomized episodes for the before / after comparison (same seeds both times)
RANGES = {"mass": (0.8, 1.2), "friction": (0.5, 1.25), "motor": (0.8, 1.2), "push": (0.0, 0.5)}


def sample_rand(seed):
    # Domain randomization (Tan et al. 2018; Rudin et al. 2021): one draw per episode
    r = np.random.default_rng(seed)
    out = {k: float(r.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}
    out["seed"] = seed
    return out


def run(tuner, train, rand=None):
    tuner.train_actor = train
    stats, tuner = run_episode(tuner=tuner, headless=True, run_t=EP_T, rand=rand)
    fell = stats["fell"] is not None
    d = np.array(tuner.log["delta"]) if tuner.log["delta"] else np.zeros(1)
    return tuner, fell, tuner.ep_cost(), float(np.sqrt(np.mean(d ** 2)))


def main():
    tuner = RLTuner(FOSMC(18, 0.002))       # layout and baseline gains only; each episode attaches its own FOSMC

    # 1. Baseline (actor at zero, no noise): nominal floor, then the fixed randomized set
    tuner, fell, c_base, _ = run(tuner, train=False)        # also runs the critic warm-up
    th_base = tuner.ep_theta()
    base_r = [run(tuner, train=False, rand=sample_rand(1000 + i)) for i in range(N_TEST)]
    print(f"[baseline] nominal cost={c_base:.3f} fell={fell}   randomized cost="
          f"{np.mean([b[2] for b in base_r]):.3f} falls={sum(b[1] for b in base_r)}/{N_TEST}", flush=True)

    # 2. Training on randomized episodes, saved after each one
    for ep in range(1, N_EPISODES + 1):
        rnd = sample_rand(ep)
        tuner, fell, c, td = run(tuner, train=True, rand=rnd)
        print(f"[train {ep:02d}] fell={fell} cost={c:.3f} td rms={td:.3f}   "
              f"mass x{rnd['mass']:.2f} mu {rnd['friction']:.2f} motor x{rnd['motor']:.2f} push {rnd['push']:.2f}",
              flush=True)
        tuner.save(SAVE_PATH)

    # 3. Evaluation (learned actor, no noise, nothing learns): same nominal + randomized set as the baseline
    tuner, fell, c_eval, _ = run(tuner, train=False)
    th_eval = tuner.ep_theta()
    eval_r = [run(tuner, train=False, rand=sample_rand(1000 + i)) for i in range(N_TEST)]
    print(f"[eval] nominal cost={c_eval:.3f} fell={fell} (baseline {c_base:.3f})   randomized cost="
          f"{np.mean([e[2] for e in eval_r]):.3f} falls={sum(e[1] for e in eval_r)}/{N_TEST} "
          f"(baseline {np.mean([b[2] for b in base_r]):.3f})", flush=True)
    print("mean gains over the nominal episode (baseline -> learned):")
    for i, name in enumerate(tuner.names):
        print(f"  {name:<14} {th_base[i]:8.3f} -> {th_eval[i]:8.3f}")

    # 4. Save / load check (nothing learns during evaluation, so the file must match)
    t2 = RLTuner(FOSMC(18, 0.002))
    t2.load(SAVE_PATH)
    same = np.allclose(t2.actor.w, tuner.actor.w) and np.allclose(t2.critic.w, tuner.critic.w) # type: ignore
    print(f"[save] {SAVE_PATH}   reload matches: {same}", flush=True)


if __name__ == "__main__":
    main()
