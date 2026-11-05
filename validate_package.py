import importlib
libs = ["numpy", "scipy", "mujoco", "casadi", "adam", "pinocchio",
        "gymnasium", "torch", "stable_baselines3", "tensorboard"]
for name in libs:
    try:
        mod = importlib.import_module(name)
        print(f"ok       {name:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"MISSING  {name:18s} {type(e).__name__}: {e}")

# the QP solvers the WBC, the force QP and the MPC use (bundled with casadi)
import casadi as cs
for solver in ("qpoases", "osqp"):
    print(f"{'ok' if cs.has_conic(solver) else 'MISSING':8s} casadi {solver}")

import torch
print("torch cpu-only build" if "+cpu" in torch.__version__ else f"torch build {torch.__version__}, cuda available: {torch.cuda.is_available()}")
