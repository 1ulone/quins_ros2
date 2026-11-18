import os
import sys
import threading
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from HARDWARE.GaitPlayback import IK_TOL_M, KNEE_DIR, STRIDE_RAMP_S
from LOGIC.GaitPlanner import LEG_NAMES, GAIT_TABLE, GaitPlanner, StandPlanner 
from LOGIC.KinematicsLogic import KinematicsLogic

GAIT = "WALK"

LEG_ID = {
    "FL": ("/dev/ttyUSB0", [11, 12, 13]),
    "FR": ("/dev/ttyUSB1", [21, 22, 23]),
    "BL": ("/dev/ttyUSB2", [31, 32, 33]),
    "BR": ("/dev/ttyUSB3", [41, 42, 43]),
}

STANCE = {leg: np.array([0.0, -1.2, 1.7]) for leg in LEG_NAMES}
SIGN = {
    "FL": np.array([-1.0, 1.0, 1.0]),
    "FR": np.array([1.0,-1.0,-1.0]),
    "FL": np.array([-1.0, 1.0, 1.0]),
    "FR": np.array([1.0,-1.0,-1.0]),
}

Q_LO = np.array([-np.pi + 0.05, -np.pi + 0.05, 0.838])
Q_HI = np.array([np.pi - 0.05, np.pi - 0.05, 2.723])

KNEE_DIR = -1
RATE_HZ = 100.0
TIME_SCALE = 0.25
STANCE_CHECK_RAD = 0.01
STRIDE_RAMP_S = 1.0
START_SPEED = 0.1

def urdf_to_motor(leg, q):
    return SIGN[leg] * q 

def step_toward(current, desired, max_step):
    return current + np.clip(desired - current, -max_step, max_step)

class HardwareCore:
    def __init__(self, drivers):
        script_dir = Path(__file__).resolve().parent
        urdf_path = str(script_dir.parent/'urdf'/'quadruped.urdf')

        self.drivers = drivers

        self.kin = KinematicsLogic(urdf_path)
        self.stand_planner = StandPlanner

        self.lock = threading.Lock()
        self.state = "IDLE"
        self.foot_home = {}
        for leg in LEG_NAMES:
            stance = STANCE[leg]
            self.foot_home[leg] = self.kin.fk(leg, *np.degrees(stance))[0:3, 3]
            q_check = np.array(self.kin.ik(leg, *self.foot_home[leg], knee_dir=-1))
            if np.max(np.abs(q_check - stance)) > STANCE_CHECK_RAD:
                raise RuntimeError(f"{leg}: IK of the stance xyz gives {np.round(q_check, 3)}, not {stance}")

        self.plan = GaitPlanner(replace(GAIT_TABLE[GAIT]), {leg: self.foot_home[leg][0:2] for leg in LEG_NAMES}) 

        self.utime = 0.0
        self.stop_flag = threading.Event()
        self.thread = None

    def leg_angles(self, leg, t, q_prev, stride_scale):
        self.plan.stride_scale = stride_scale
        foot_xyz = self.foot_home[leg] + np.array(self.plan.foot_offset(leg, t))
        q = np.array(self.kin.ik(leg, *foot_xyz, knee_dir=KNEE_DIR))
        foot_reached = self.kin.fk(leg, *np.degrees(q))[0:3, 3]
        if np.linalg.norm(foot_reached - foot_xyz) > IK_TOL_M:
            return q_prev
        return np.clip(q, Q_LO, Q_HI)

    def start(self, time_scale=TIME_SCALE):
        if self.thread is not None and self.thread.is_alive():
            return
