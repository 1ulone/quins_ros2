import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from LOGIC.KinematicsLogic import KinematicsLogic
from LOGIC.GaitPlanner import GaitPlanner, GAIT_TABLE
from LOGIC.ControllerCore import NOMINAL_STANCE, WALK_DROP

LEG = "FL"              # leg to evaluate (FL / FR / BL / BR)
GAIT = "WALK"           # entry in GAIT_TABLE
N_SAMPLES = 400         # samples over one gait cycle

kin = KinematicsLogic(os.path.join(ROOT, "urdf", "quadruped.urdf"))
leg_i = ["FL", "FR", "BL", "BR"].index(LEG)
q_nom = NOMINAL_STANCE[3 * leg_i: 3 * leg_i + 3]

# walk pose: same as TaskLogic._lowered_pose(), foot at the same x, y but WALK_DROP closer to the body
foot_nom = kin.fk(LEG, *np.degrees(q_nom))[0:3, 3]          # fk takes degrees
knee_dir = -int(np.sign(q_nom[2]))                          # same knee side as the nominal pose
foot_home = foot_nom + np.array([0.0, 0.0, WALK_DROP])
q_walk = np.array(kin.ik(LEG, *foot_home, knee_dir=knee_dir))

# one gait cycle: foot = walk-pose foot + planner offset (stance sweep + swing lift)
planner = GaitPlanner(GAIT_TABLE[GAIT], {LEG: foot_home[0:2]})
cycle_s = 1.0 / planner.p.freq
q_cycle = []
for t in np.linspace(0.0, cycle_s, N_SAMPLES, endpoint=False):
    off = np.array(planner.foot_offset(LEG, t))
    q_cycle.append(kin.ik(LEG, *(foot_home + off), knee_dir=knee_dir))
q_cycle = np.array(q_cycle)

print(f"leg {LEG}, gait {GAIT}, one cycle = {cycle_s:.3f} s")
print(f"nominal stance (rad): {np.round(q_nom, 4)}")
print(f"walk pose      (rad): {np.round(q_walk, 4)}")
for j, name in enumerate(["HAA", "HFE", "KFE"]):
    lo, hi = q_cycle[:, j].min(), q_cycle[:, j].max()
    print(f"{name}: min {lo:+.4f}   max {hi:+.4f}   range {hi - lo:.4f} rad")
