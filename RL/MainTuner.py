import numpy as np
from LOGIC.FOSMCLogic import GROUPS, GAIN_NAMES
from RL.Critic import RBFCritic, g_cost
from RL.Actor import RBFActor

# NOTE: 14 variables that is selected to be tune are :
# Ke1, Ke2, Ks, Kr, for base_lin, base_ang, swing, then lambda, alpha


# Ke1 / Ks bounds at 0.75x .. 1.5x baseline, both checked under RAND_EDGE (2x upper fell in a randomized eval).
# Ke2 and Kr lower bounds at 0.5x baseline: Farid requires both positive definite (eq. 31, eq. 48)
GAIN_BOUNDS = {
    "base_lin": {
        "Ke1": (6.0, 12.0),
        "Ke2": (0.5, 1.5),
        "Ks": (6.0, 12.0),
        "Kr": (0.25, 1.0)
    },
    "base_ang": {
        "Ke1": (10.5, 21.0),
        "Ke2": (0.5, 1.5),
        "Ks": (10.5, 21.0),
        "Kr": (0.5, 2.0)
    },
    "swing": {
        "Ke1": (22.5, 45.0),
        "Ke2": (0.375, 1.0),
        "Ks": (22.5, 45.0),
        "Kr": (1.0, 4.0)
    },
}

LAM_BOUNDS =  (0.95, 1.0)
ALPHA_BOUNDS = (1.25, 1.75)

# NOTE: personal note: read this later pls 

# Cost (Farid eq. 35) per gain group, set from the baseline run (RL_TEST = None):
# eps = p25 of |s| (dead zone: no cost, no learning), xi = p99 of |s| (cost cap)
S_EPS = {"base_lin": 0.043, "base_ang": 0.032, "swing": 0.052}   # p25 of the fixed-reference s (baseline run)
S_XI = {"base_lin": 0.354, "base_ang": 0.262, "swing": 0.681}    # p99 of the fixed-reference s (baseline run)
TAU2_REF = 472.2    # baseline mean ||tau||^2, makes the torque term unitless
W_TAU = 0.3         # torque term weight (starting value, tie-breaker next to tracking)
W_DELTA = 1.0       # gain change weight, on gains scaled to [0, 1] (starting value)
GAMMA = 0.98        # per RL step: horizon 1 / (1 - gamma) = 50 steps = 1 s, about 2.6 gait cycles
WARMUP = 200        # RL steps (4 s) of walking data before the critic starts
LOG_R0 = 0.1        # critic input ln(r + 0.1); 0.1 ~ the dead zone (eps / xi), keeps the log finite at r = 0
N_LOG = 5           # grid levels per error input, covering r = 0 .. 3 (r = rms(s_bar / xi): 1 = at the cost cap)
LOG_LO, LOG_HI = np.log(LOG_R0), np.log(3.0 + LOG_R0)
STEP_LOG = (LOG_HI - LOG_LO) / (N_LOG - 1)
N_PHASE = 8         # grid points around the gait-phase circle
RAD_PHASE = 1.0 / (2.0 * np.sin(np.pi / N_PHASE))     # circle radius that gives unit spacing between phase points
LR_ACTOR = 0.05     # actor step, below the critic floor (0.02) so the critic keeps up (two time-scales)
EXPLORE = 0.1       # exploration std in pre-squash units: ~2.5% of each gain's range near the center
LEAK_ACTOR = 1e-3   # leakage: pull toward the pretrained actor (zero = baseline gains until piece 4)
U_SHIFT_MAX = 3.0   # cap on the learned shift in pre-squash units
HOLD = 25


def _layout(fosmc):
    names, lo, hi, base = [], [], [], []
    for g in GROUPS:
        for k in GAIN_NAMES:
            names.append(f"{g}.{k}")
            lo.append(GAIN_BOUNDS[g][k][0])
            hi.append(GAIN_BOUNDS[g][k][1])
            base.append(fosmc.groups[g][k])

    names += ["lam", "alpha"]
    lo += [LAM_BOUNDS[0], ALPHA_BOUNDS[0]]
    hi += [LAM_BOUNDS[1], ALPHA_BOUNDS[1]]
    base += [fosmc.lam, fosmc.alpha]

    return names, np.array(lo), np.array(hi), np.array(base)

def _grid():
    # RBF units on a fixed grid with unit spacing: 5 x 5 x 5 error levels x 8 phase points = 1000 units
    lv = np.arange(N_LOG, dtype=float)
    ang = 2.0 * np.pi * np.arange(N_PHASE) / N_PHASE
    ph = RAD_PHASE * np.stack([np.sin(ang), np.cos(ang)], axis=1)
    return np.array([[a, b, c, p[0], p[1]] for a in lv for b in lv for c in lv for p in ph])

class RLTuner:
    # NOTE: Actor-Critic Gain Tuner for the task-space FOSMC as per Farid Yousuf Paper

    def __init__(self, fosmc, test=None):
        self.fosmc = fosmc
        self.test = test
        self.names, self.lo, self.hi, self.base = _layout(fosmc)

        # NOTE: pre meal prep or something, give baseline gains
        frac0 = np.clip((self.base - self.lo) / (self.hi - self.lo), 0.02, 0.98)
        self.u0 = np.log(frac0 / (1.0 - frac0))

        self.theta = self.squash(self.u0)
        self.theta_prev = self.theta.copy()
        self.apply(self.theta)
        self.k_ref = fosmc.Ke1.copy()

        self.last_dbg = None
        self.n_obs = 0
        self.rec = {"s":{g: [] for g in GROUPS}, "tau2": [], "steps": 0}
        self.eps, self.xi = np.zeros(fosmc.dof), np.ones(fosmc.dof)
        for g, rows in GROUPS.items():
            self.eps[rows], self.xi[rows] = S_EPS[g], S_XI[g]
        # a fall counts as the worst cost (every row at its cap) for the rest of the horizon
        self.j_fall = float(self.xi.sum()) / (1.0 - GAMMA)
        self.acc = {"g": 0.0, "tau": 0.0, "ks": 0.0, "n": 0}
        self.warm, self.warm_cost, self.critic, self.z_prev = [], [], None, None
        self.log = {"delta": [], "cost": [], "J": [], "act": []}
        self.actor = None
        self.eps_a = None
        self.d_theta = np.zeros(len(self.u0)) 
        self.train_actor = True
        self.ep = {"cost": 0.0, "n": 0, "theta": np.zeros(len(self.u0)), "k": 0}
        self.win = None
        self.hold = 9
        self.u_win = None

    def squash(self, u):
        if self.test == "min":
            return self.lo.copy()
        if self.test == "max":
            return self.hi.copy()
        if self.test == "base":
            u = self.u0
        return self.lo + (self.hi - self.lo) / (1.0 + np.exp(-u))

    def apply(self, theta):
        groups, i = {}, 0

        for g in GROUPS:
            groups[g] = {k: float(theta[i + j]) for j, k in enumerate(GAIN_NAMES)}
            i += len(GAIN_NAMES)

        self.fosmc.update_gains(groups=groups, lam=float(theta[i]), alpha=float(theta[i+1]))

    def observe(self, dbg, tau, record=False):
        # NOTE: called on every WBC step
        self.last_dbg = dbg

        if not record:
            return

        act = dbg["active"]
        s_all = dbg["e_dot"] + self.k_ref * dbg["e"]
        for g, rows in GROUPS.items():
            a = act[rows]
            if np.any(a):
                self.rec["s"][g].extend(np.abs(s_all[rows][a]).tolist())

        self.rec["tau2"].append(float(tau @ tau))
        # interval cost terms: eq. 35 summed over the active rows, torque, ||s|| for eq. 42
        s = s_all[act]
        self.acc["g"] += float(np.sum(g_cost(s, self.eps[act], self.xi[act])))
        self.acc["tau"] += float(tau @ tau) / TAU2_REF
        self.acc["ks"] += float(np.sqrt(np.mean((s / self.xi[act]) ** 2)))
        self.acc["n"] += 1
        self.ep["cost"] += float(np.sum(g_cost(s, self.eps[act], self.xi[act]))) + W_TAU * float(tau @ tau) / TAU2_REF
        self.ep["n"] += 1

    def observation(self):
        d = self.last_dbg
        e, ed, act = d["e"], d["e_dot"], d["active"] # type: ignore
        s = ed + self.k_ref * e
        sw = act[6:]
        sw_stats = [float(np.mean(np.abs(v[6:][sw]))) if np.any(sw) else 0.0 for v in (s, e, ed)]
        ph = 2.0 * np.pi * d["phase"] # type: ignore
        th = 2.0 * (self.theta - self.lo) / (self.hi - self.lo) - 1.0
        return np.concatenate([e[0:6], ed[0:6], s[0:6], sw_stats, [np.sin(ph), np.cos(ph)], th])

    def critic_input(self):
        # 5 numbers that mean the same under any gains or conditions: per group, the size of the
        # fixed-reference error (rms of s_bar / xi, log scale, in grid units), plus the gait phase
        d = self.last_dbg
        act = d["active"]  # type: ignore
        s_bar = (d["e_dot"] + self.k_ref * d["e"]) / self.xi  # type: ignore
        z = []
        for rows in GROUPS.values():
            a = act[rows]
            r = float(np.sqrt(np.mean(s_bar[rows][a] ** 2))) if np.any(a) else 0.0
            z.append(float(np.clip((np.log(r + LOG_R0) - LOG_LO) / STEP_LOG, 0.0, N_LOG - 1)))
        ph = 2.0 * np.pi * d["phase"]  # type: ignore
        z += [RAD_PHASE * np.sin(ph), RAD_PHASE * np.cos(ph)]
        return np.array(z)

    def step(self, fell=False):
        if self.last_dbg is None:
            return
        obs = self.observation()
        self.n_obs = obs.size
        z = self._learn(obs, fell)          # grade the interval that just ended first
        u = self.u0.copy()
        if self.actor is not None and z is not None and not fell:
            if self.train_actor:
                if self.hold == 0:
                    # new exploration window: one noise sample held for HOLD steps, so its effect
                    # on the cost lasts long enough to be measured
                    self.u_win, self.eps_a = self.actor.act(z, self.u0)
                    self.win = {"z": z, "eps": self.eps_a, "ret": 0.0, "k": 0, "live": 0}
                    self.hold = HOLD
                self.hold -= 1
                u = self.u_win
            else:
                u = self.u0 + self.actor.shift(z)             # evaluation: learned shift only, no noise
        else:
            self.win, self.hold = None, 0

        theta = self.squash(u)
        self.d_theta = (theta - self.theta) / (self.hi - self.lo)   # charged to the next interval's cost
        self.theta_prev, self.theta = self.theta, theta
        self.apply(theta)
        self.rec["steps"] += 1
        if z is not None:
            self.ep["theta"] += theta
            self.ep["k"] += 1

    def _learn(self, obs, fell):
        # close the interval: mean cost per WBC step over the last T
        n = self.acc["n"]
        cost, k_s = 0.0, 1.0
        if n > 0:
            cost = (self.acc["g"] + W_TAU * self.acc["tau"]) / n + W_DELTA * float(self.d_theta @ self.d_theta)
            k_s = min(self.acc["ks"] / n, 1.0)
        dead = self.acc["g"] == 0.0
        self.acc = {"g": 0.0, "tau": 0.0, "ks": 0.0, "n": 0}
        if n == 0 and not fell:
            self.z_prev = None          # not walking: no transition to learn from
            return
        if fell and self.z_prev is None:
            return

        z = self.critic_input()
        if self.critic is None:
            # warm-up: only the cost level is measured here; the RBF units sit on a fixed grid
            self.warm.append(z)
            self.warm_cost.append(cost)
            if len(self.warm) >= WARMUP:
                self.critic = RBFCritic(_grid(), gamma=GAMMA)
                self.critic.w[-1] = float(np.mean(self.warm_cost)) / (1.0 - GAMMA)
                self.actor = RBFActor(self.critic, len(self.u0), lr=LR_ACTOR, explore=EXPLORE, leak=LEAK_ACTOR, shift_max=U_SHIFT_MAX)
                act = np.array([self.critic.features(zz)[:-1].max() for zz in self.warm])
                print(f"[critic] started on a {len(self.critic.c)}-unit grid, rbf width = {self.critic.b:.2f}   "
                      f"warm-up max act = {act.mean():.3f}", flush=True)
            return
        if self.z_prev is not None:
            j_prev = self.critic.value(self.z_prev)
            if fell:
                delta = self.critic.update(self.z_prev, cost, None, k_s=1.0, terminal_value=self.j_fall, learn=self.train_actor)
            else:
                delta = self.critic.update(self.z_prev, cost, z, k_s=1.0, learn=self.train_actor)
            self.log["delta"].append(delta)
            self.log["cost"].append(cost)
            self.log["J"].append(j_prev)
            self.log["act"].append(float(self.critic.features(self.z_prev)[:-1].max()))   # strongest RBF unit (coverage check)

            if self.win is not None and self.actor is not None:
                # n-step actor update: grade the whole exploration window, not one 20 ms step
                w = self.win
                w["ret"] += GAMMA ** w["k"] * cost
                w["live"] += 0 if dead else 1
                w["k"] += 1
                if w["k"] >= HOLD or fell:
                    j_end = self.j_fall if fell else self.critic.value(z)
                    adv = self.critic.value(w["z"]) - (w["ret"] + GAMMA ** w["k"] * j_end)   # > 0: cheaper than expected
                    if w["live"] > 0:           # dead zone: skip windows that stayed inside eps the whole time
                        self.actor.update(w["z"], adv, w["eps"])
                    self.win = None

            if len(self.log["delta"]) % 250 == 0:
                d = np.array(self.log["delta"][-250:])
                print(f"[critic] steps={len(self.log['delta'])} td rms={np.sqrt(np.mean(d ** 2)):.3f} "
                      f"cost={np.mean(self.log['cost'][-250:]):.3f} J={j_prev:.2f} "
                      f"max act={np.mean(self.log['act'][-250:]):.3f}", flush=True)
        self.z_prev = None if fell else z
        return None if fell else z

    def summary(self):
        print("\n===== rl tuner (piece 1) =====", flush=True)
        print(f"obs size            : {self.n_obs}")
        print(f"rl steps            : {self.rec['steps']}")
        print(f"test mode           : {self.test}")
        print(f"lam / alpha         : {self.fosmc.lam:.4f} / {self.fosmc.alpha:.4f}")
        cur = np.array([self.fosmc.groups[g][k] for g in GROUPS for k in GAIN_NAMES])
        print(f"max gain deviation  : {np.max(np.abs(cur - self.base[:-2])):.4f}")
        for g in GROUPS:
            v = np.array(self.rec["s"][g])
            if v.size:
                p = np.percentile(v, [25, 50, 90, 99])
                print(f"{g:<9} |s| p25/p50/p90/p99 = " + "/".join(f"{x:.3f}" for x in p))
        if self.rec["tau2"]:
            t2 = np.array(self.rec["tau2"])
            print(f"||tau||^2 mean / p99 = {t2.mean():.1f} / {np.percentile(t2, 99):.1f}", flush=True)
        if len(self.log["delta"]) > 600:
            c = np.array(self.log["cost"])
            J = np.array(self.log["J"])
            d = np.array(self.log["delta"])
            G, run = np.zeros_like(c), 0.0
            for k in range(len(c) - 1, -1, -1):     # discounted cost-to-go actually received
                run = c[k] + GAMMA * run
                G[k] = run
            end = len(c) - 150                      # drop the tail, where G is cut short
            a, b = end // 3, 2 * end // 3
            Gl, Jl = G[b:end], J[b:end]
            print(f"td rms first / last third : {np.sqrt(np.mean(d[:a] ** 2)):.3f} / {np.sqrt(np.mean(d[b:end] ** 2)):.3f}"
                  f"   (constant-J level = std of cost: {c[b:end].std():.3f})")
            print(f"critic last third         : corr(J, G) = {np.corrcoef(Gl, Jl)[0, 1]:.2f}   "
                  f"rms(J - G) / mean(G) = {np.sqrt(np.mean((Gl - Jl) ** 2)) / (abs(Gl.mean()) + 1e-9):.3f}   "
                  f"mean J / G = {Jl.mean():.2f} / {Gl.mean():.2f}   std G = {Gl.std():.2f}", flush=True)
            n10 = max(len(c) // 10, 1)
            print(f"cost per step first / last 10% : {c[:n10].mean():.3f} / {c[-n10:].mean():.3f}")
        th_mean = self.ep_theta() if self.ep["k"] > 0 else self.theta
        print("gains (baseline -> mean over the run) [bounds]:")    # the last step resets to baseline after a fall
        for i, name in enumerate(self.names):
            print(f"  {name:<14} {self.base[i]:8.3f} -> {th_mean[i]:8.3f}   [{self.lo[i]:.2f}, {self.hi[i]:.2f}]")

    # ---- episodes (pretraining) ----
    def attach(self, fosmc):
        # new episode: new FOSMC, same critic and actor; per-episode state back to the start
        self.fosmc = fosmc
        self.theta = self.squash(self.u0)
        self.theta_prev = self.theta.copy()
        self.apply(self.theta)
        self.last_dbg, self.z_prev, self.eps_a = None, None, None
        self.win, self.hold, self.u_win = None, 0, None
        self.d_theta = np.zeros(len(self.u0))
        self.acc = {"g": 0.0, "tau": 0.0, "ks": 0.0, "n": 0}
        self.rec = {"s": {g: [] for g in GROUPS}, "tau2": [], "steps": 0}
        self.log = {"delta": [], "cost": [], "J": [], "act": []}
        self.ep = {"cost": 0.0, "n": 0, "theta": np.zeros(len(self.u0)), "k": 0}

    def ep_cost(self):
        return self.ep["cost"] / max(self.ep["n"], 1)      # mean cost per WBC step (eq. 35 + torque)

    def ep_theta(self):
        return self.ep["theta"] / max(self.ep["k"], 1)     # mean gains over the episode

    def save(self, path):
        np.savez(path, c=self.critic.c, w=self.critic.w, n=self.critic.n, Wa=self.actor.w) # type: ignore

    def load(self, path):
        d = np.load(path)
        self.critic = RBFCritic(d["c"], gamma=GAMMA)
        self.critic.w, self.critic.n = d["w"].copy(), 0
        print(f"[load] critic n = {self.critic.n} (step restarts at {self.critic.lr}), rbf width = {self.critic.b:.2f}", flush=True)
        self.actor = RBFActor(self.critic, len(self.u0), lr=LR_ACTOR, explore=EXPLORE,
                              leak=LEAK_ACTOR, shift_max=U_SHIFT_MAX)
        self.actor.w, self.actor.w0 = d["Wa"].copy(), d["Wa"].copy()   # leakage now pulls toward the pretrained actor

