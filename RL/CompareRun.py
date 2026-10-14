import os
os.environ.setdefault("OMP_NUM_THREADS", "1")        # one core per run, so parallel runs don't fight over cores
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import multiprocessing as mp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

SEEDS = [0, 1, 2, 3, 4]                              # push patterns: each seed is one A/B pair
MODES = ("A", "C", "D")    # A fixed gains | B online (current) | C pretrained, frozen | D online, half exploration
RUN_T = 60.0
WORKERS = max(1, min(len(MODES) * len(SEEDS), (os.cpu_count() or 2) - 1))   # leave one core free
LOAD = os.path.join(current_dir, "tuner_pretrained.npz")


def one_run(job):
    mode, seed = job
    from LOGIC.FOSMCLogic import FOSMC
    from MUJOCO.MujocoSim import main as run_episode, RAND_MID, SEQ_MIXED
    from RL.MainTuner import RLTuner

    rand = dict(RAND_MID, seed=seed)                 # same mismatch, different push pattern per seed
    if mode == "A":
        tuner = RLTuner(FOSMC(18, 0.002), test="base")   # A: fixed baseline gains
    else:
        tuner = RLTuner(FOSMC(18, 0.002))
        tuner.load(LOAD)                                 # pretrained critic and actor
        tuner.train_actor = mode != "C"                  # C: frozen pretrained policy, no exploration, no learning
        if mode == "D":
            tuner.actor.explore *= 0.5 # type: ignore
    stats, tuner = run_episode(tuner=tuner, headless=True, run_t=RUN_T, rand=rand, script=SEQ_MIXED)

    n = max(stats["n"], 1)
    c = np.array(tuner.log["cost"]) if tuner.log["cost"] else np.zeros(1)
    k = max(len(c) // 10, 1)
    t2 = np.array(tuner.rec["tau2"]) if tuner.rec["tau2"] else np.zeros(1)
    return {
        "mode": mode, "seed": seed, "fell": stats["fell"],
        "cost": tuner.ep_cost(),                     # mean cost per step (tracking + torque, no exploration term)
        "last10": float(c[-k:].mean()),
        "speed": float(np.sqrt(stats["v_err2"] / n)),
        "height": float(100 * np.sqrt(stats["z_err2"] / n)),
        "late": 100 * stats["late"] / max(stats["n_st"], 1),
        "early": 100 * stats["early"] / max(stats["n_sw"], 1),
        "tau_p99": float(np.percentile(t2, 99)),
    }


def main():
    jobs = [(m, s) for s in SEEDS for m in MODES]
    print(f"[compare] {len(jobs)} runs on {WORKERS} workers, {RUN_T:.0f} s sim each", flush=True)
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        res = pool.map(one_run, jobs)                # keeps job order: A then B for each seed

    keys = ["cost", "last10", "speed", "height", "late", "early", "tau_p99"]
    print("\nseed mode  fell   cost  last10  speed  height  late%  early%  tau_p99")
    for r in res:
        fell = "no" if r["fell"] is None else f"{r['fell']:.0f}s"
        print(f"{r['seed']:>4} {r['mode']:>4} {fell:>5} {r['cost']:6.3f} {r['last10']:7.3f} {r['speed']:6.3f} "
              f"{r['height']:7.2f} {r['late']:6.1f} {r['early']:7.1f} {r['tau_p99']:8.1f}")

    print("\nmean over seeds (A = fixed gains, B = RL tuner):")
    for m in MODES:
        rs = [r for r in res if r["mode"] == m]
        falls = sum(r["fell"] is not None for r in rs)
        print(f"  {m}: falls {falls}/{len(rs)}  " + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.3f}" for k in keys))
    a = np.array([r["cost"] for r in res if r["mode"] == "A"])
    for m in MODES[1:]:
        x = np.array([r["cost"] for r in res if r["mode"] == m])
        print(f"  {m} vs A cost: {100 * (x.mean() / a.mean() - 1):+.1f} %   (lower in {int(np.sum(x < a))}/{len(a)} seeds)", flush=True)


if __name__ == "__main__":
    main()
