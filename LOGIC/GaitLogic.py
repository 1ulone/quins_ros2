from dataclasses import replace
import math as m
import numpy as np
from LOGIC.GaitPlanner import GAIT_TABLE, GaitPlanner, JumpParams, StandPlanner, JumpPlanner
from LOGIC.KinematicsLogic import KinematicsLogic
from pathlib import Path

# NOTE: ---------------- CONSTANT VALUE ----------------
LEG_NAMES = [
    'FL',
    'FR',
    'BL',
    'BR'
]

JOINT_NAMES = {
    'FL': ['LF_HAA', 'LF_HFE', 'LF_KFE'],
    'FR': ['RF_HAA', 'RF_HFE', 'RF_KFE'],
    'BL': ['LH_HAA', 'LH_HFE', 'LH_KFE'],
    'BR': ['RH_HAA', 'RH_HFE', 'RH_KFE'],
}

class GaitLogic():
    def __init__(self, callbacks=None):
        script_dir = Path(__file__).resolve().parent
        urdf_path = str(script_dir.parent/'urdf'/'quadruped.urdf')

        self.kinematics = KinematicsLogic(urdf_path)
        self.planner = None
        self.stand_planner = StandPlanner()
        self.state_start_time = None
        self.sim_time = 0.0
        self.nominal_feet = {leg: self.kinematics.get_init_pos(leg)[0:2] for leg in LEG_NAMES}
        self.control_rate = 50.0 
        self.robot_mass = 10.2

        self.current_yaw = 0.0
        self.target_yaw = 0.0

        self.walking = False
        self.turning = False

        self.pending_state = None
        self.half_cycle = None
        self.settle_until = None
        self.ready_at = 0.0

        self.current_state = "TUNING"

    # NOTE: -------- Callback --------
    def request_state(self, msg: str):
        # NOTE: GUI thread only stores the request; loop_step applies it on the sim clock
        self.pending_state = msg

    def _apply_pending(self, sim_time):
        msg = self.pending_state
        if msg is None or self.settle_until is not None or sim_time < self.ready_at:
            return      # nothing asked, or still re-posing / settling in IDLE: the request waits
        # NOTE: a running gait only switches when all four feet are down; with the 0.5 duty
        # trot that is every half cycle (one pair lands as the other lifts)
        if isinstance(self.planner, GaitPlanner) and self.state_start_time is not None:
            half = m.floor(2.0 * (sim_time - self.state_start_time) * self.planner.p.freq)
            if self.half_cycle is None:
                self.half_cycle = half
            if half == self.half_cycle:
                return
        self.half_cycle = None
        if msg != "IDLE" and isinstance(self.planner, GaitPlanner):
            self.update_state("IDLE")   # every change passes through IDLE: re-pose first, the request stays pending
            return
        self.pending_state = None
        self.update_state(msg)

    def update_state(self, msg: str):
        # NOTE: if state_callback is triggered again, 
        # it will cancel the walk timer, to avoid duplicated process
        old = self.planner if isinstance(self.planner, GaitPlanner) else None
        old_t0 = self.state_start_time
        self.settle_until = None

        self.walking = False
        self.turning = False
        self.yaw_rate = 0.0
        self.current_state = msg

        self.state_start_time = None
        self.planner = None

        match msg:
            case "IDLE":
                if old is not None and old_t0 is not None:
                    self.planner = GaitPlanner(replace(GAIT_TABLE["WALK"], step_len=0.0), self.nominal_feet)
                    p = self.planner.p
                    c = ((self.sim_time - old_t0) * old.p.freq + old.p.offsets['FL']) % 1.0   # FL phase at the switch
                    self.state_start_time = self.sim_time - ((c - p.offsets['FL']) % 1.0) / p.freq
                    self.settle_until = self.sim_time + 1.0 / p.freq
            case "WALK" | "WALK_BACK" | "CRAWL" | "RUN" | "TURN" | "TURN_RIGHT":
                gait = {"WALK_BACK": "WALK", "TURN_RIGHT": "TURN"}.get(msg, msg)
                self.planner = GaitPlanner(replace(GAIT_TABLE[gait]), self.nominal_feet)
                if msg == "WALK_BACK":
                    self.planner.p.step_len = -self.planner.p.step_len
                if msg == "TURN_RIGHT":
                    self.planner.p.sc_yaw = -self.planner.p.sc_yaw

                self.walking = gait != "TURN"
                self.turning = gait == "TURN"
                self.target_yaw = self.current_yaw

    def gait_process(self, t):
        # NOTE: WALK | CRAWL | RUN | TURN uses this process
        planner = self.planner
        assert isinstance(planner, GaitPlanner)
        p = planner.p

        # NOTE: 1 second startup transition blend
        ramp = min(t / 1.0, 1.0)
        planner.stride_scale = ramp

        # NOTE: on turning
        if self.turning:
            if abs(self.yaw_rate) < 0.1:
                self.yaw_rate = p.sc_yaw
            planner.yaw_rate = self.yaw_rate

    def loop_step(self, sim_time):
        self.sim_time = sim_time
        if self.settle_until is not None and sim_time >= self.settle_until:
            self.settle_until = None
            self.planner = None                     # re-posed: stand on all four feet
            self.state_start_time = sim_time
            self.ready_at = sim_time + 0.25         # short stand so the last pair takes the load before the next gait
        self._apply_pending(sim_time)
        if self.state_start_time is None:
            self.state_start_time = sim_time

        t = sim_time - self.state_start_time
        if self.walking or self.turning:
            self.gait_process(t)

    def wbc_targets(self, sim_time):
        # NOTE: Single-instant schedule for the WBC; anything that is not a periodic gait
        # (transitions, jump) stands on all four feet for now
        if self.state_start_time is None or not isinstance(self.planner, GaitPlanner):
            return self.stand_planner.instant(0.0, self.robot_mass)
        return self.planner.instant(sim_time - self.state_start_time, self.robot_mass)

