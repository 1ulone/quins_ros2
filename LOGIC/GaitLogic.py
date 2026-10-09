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
        self.x_off_used, self.z_off_used = 0.0, 0.39
        self.start_x_off, self.start_z_off = 0.0, 0.39

        self.callbacks = callbacks if callbacks else {}

        self.control_rate = 50.0 
        self.dt = 1.0 / self.control_rate # Time step (0.02s)
        self.t = 0.0 # global timer (s) incremented everey walk_process cycle (0 on enter)
        self.graph_t = 0.0
        self.robot_mass = 10.2

        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.yaw_rate = 0.0

        # NOTE: Inverse Dynamics Parameters (Velocity & Acceleration)
        self.current_q = np.zeros(12)
        self.current_q_dot = np.zeros(12)
        self.filtered_fz = np.zeros(4)

        # NOTE: Every leg part index / tag
        self.phi = {
            leg: {
                "shoulder": 0.0,
                "thigh": 0.0,
                "leg": 0.0
            } for leg in LEG_NAMES
        }

        self.walking = False
        self.turning = False
        self.yaw_rate = 0.0
        self.jump_state = ""
        self.jump_q_history = np.zeros(12)
        self.jump_qd_history = np.zeros(12)

        self.pending_state = None
        self.half_cycle = None

        self.transitioning = False
        self.transition_elapsed = 0.0
        self.transition_duration = 1.0
        self.transition_initial = {}
        self.transition_target = (0.0, 0.0, 0.0)

        # NOTE: WALK Tune Parameters 
        self.gait_freq = 1.0
        self.x_off = 0.25
        self.z_off = 2.7
        self.step_len = 2.0
        self.step_h = 0.75
        self.sc_yaw = 0.9

        # NOTE: JUMP Tune Parameters
        self.y_crouch = 1.0
        self.y_thrust = 3.8
        self.y_flight = 1.8
        self.x_thrust = 0.5
        self.x_flight =-1.5
        self.x_catch = -1.5
        self.prepare_time = 0.8
        self.front_thrust_time = 0.15
        self.back_thrust_time = 0.15
        self.flight_time = 0.05
        self.landing_time = 0.5
        self.catch_time = 0.1
        self.x_stabilize = 1.5
        self.back_thrust = 1.5
        self.pitch_threshold = 0.0

        self.current_state = "TUNING"
        self.duty_factor = 0.5 

        # NOTE: Phase Offsets
        # 3.14 -> 360 degree, radian to angle 
        # phase offsets results in a full radian cycle 0 - 3.14
        self.phase_offsets = {
            'FL': 0.0, # (0 / 360 degree)
            'BR': 0.0, # (90 degree)
            'FR': m.pi, # (180 degree)
            'BL': m.pi, # (270 degree)
        }

    # NOTE: -------- Callback --------
    def request_state(self, msg: str):
        # NOTE: GUI thread only stores the request; loop_step applies it on the sim clock
        self.pending_state = msg

    def _apply_pending(self, sim_time):
        msg = self.pending_state
        if msg is None:
            return
        # NOTE: a running gait only switches when all four feet are down; with the 0.5 duty
        # trot that is every half cycle (one pair lands as the other lifts)
        if isinstance(self.planner, GaitPlanner) and self.state_start_time is not None:
            half = m.floor(2.0 * (sim_time - self.state_start_time) * self.planner.p.freq)
            if self.half_cycle is None:
                self.half_cycle = half
            if half == self.half_cycle:
                return
        self.pending_state = None
        self.half_cycle = None
        self.update_state(msg)
    def update_state(self, msg: str):
        # NOTE: if state_callback is triggered again, 
        # it will cancel the walk timer, to avoid duplicated process
        self.walking = False
        self.turning = False
        self.jump_state = ""
        self.transitioning = False
        self.yaw_rate = 0.0
        self.current_state = msg

        self.state_start_time = None
        self.planner = None

        match msg:
            case "TUNING":
                # NOTE: handled on GUI
                pass
            case "CROUCH":
                self.setup_transition(0.00, 1.30, -2.70)
            case "IDLE":
                self.setup_transition(0.00, 0.60, -0.90)
            case "WALK" | "WALK_BACK" | "CRAWL" | "RUN" | "TURN" | "TURN_RIGHT":
                gait = {"WALK_BACK": "WALK", "TURN_RIGHT": "TURN"}.get(msg, msg)
                self.planner = GaitPlanner(replace(GAIT_TABLE[gait]), self.nominal_feet)
                if msg == "WALK_BACK":
                    self.planner.p.step_len = -self.planner.p.step_len
                if msg == "TURN_RIGHT":
                    self.planner.p.sc_yaw = -self.planner.p.sc_yaw
                self.x_off, self.z_off = self.planner.p.x_off, self.planner.p.z_off

                self.start_x_off, self.start_z_off = self.x_off_used, self.z_off_used

                self.walking = gait != "TURN"
                self.turning = gait == "TURN"
                self.target_yaw = self.current_yaw
            case "JUMP":
                self.setup_transition(0.00, -0.45, -0.60)
                self.planner = JumpPlanner(JumpParams(
                    x_off=self.x_off, z_off=self.z_off,
                    y_crouch=self.y_crouch, y_thrust=self.y_thrust, y_flight=self.y_flight,
                    x_flight=self.x_flight, x_catch=self.x_catch,
                    x_stabilize=self.x_stabilize, back_thrust=self.back_thrust,
                    prepare_time=self.prepare_time, back_thrust_time=self.back_thrust_time,
                    flight_time=self.flight_time, catch_time=self.catch_time,
                    landing_time=self.landing_time,
                ))
                self.jump_state = "PREPARE"
                self.jump_q_history = np.copy(self.current_q)
                self.jump_qd_history = np.zeros(12)


    def update_wt_params(self, msg: list):
        # NOTE: just sets the Walk Tune Param into a new Value from msg

        self.x_off = msg[1]
        self.z_off = msg[2]
        if isinstance(self.planner, GaitPlanner):
            p = self.planner.p
            p.freq, p.x_off, p.z_off, p.step_len, p.step_h, p.sc_yaw = msg[0:6]

    def update_jt_params(self, msg: list):
        # NOTE: just sets the Jump Tune Param into a new Value from msg
        self.y_crouch = msg[0]
        self.y_thrust = msg[1] 
        self.y_flight = msg[2] 
        self.x_thrust = msg[3]
        self.x_flight = msg[4]
        self.x_catch = msg[5]
        self.prepare_time = msg[6] 
        self.front_thrust_time = msg[7] 
        self.back_thrust_time = msg[8] 
        self.flight_time = msg[9]
        self.landing_time = msg[10]
        self.catch_time = msg[11]
        self.x_stabilize = msg[12]
        self.back_thrust = msg[13]
        self.pitch_threshold = msg[14]

    def update_phase_offsets(self, msg: list):
        # NOTE: just sets the Phase Offsets into a new Value from msg
        self.phase_offsets = {
            'FL': msg[0],
            'BR': msg[1],
            'FR': msg[2],
            'BL': msg[3],
        }

        if isinstance(self.planner, GaitPlanner):
            self.planner.p.offsets = {
                leg: (msg[i] / (2.0 * m.pi)) % 1.0 for i, leg in enumerate(['FL', 'BR', 'FR', 'BL'])
            }

    def update_gamepad_params(self, msg: list):
        if not self.walking:
            return

        ly = msg[0]
        rx = msg[1]
        
        if ly == 0:
            forward_dir = 0.0
        else:
            forward_dir = 1.0 if ly < 0 else -1.0

        scale = 0.44 if self.current_state == "RUN" else 0.146
        self.step_len = forward_dir * scale 

        if isinstance(self.planner, GaitPlanner):
            self.planner.p.step_len = self.step_len

        yaw_turn_rate = 0.02
        self.target_yaw += rx * yaw_turn_rate


    def raw_tune(self, msg: list):
        # NOTE: Raw Tuning just sends a theta value from msg (per coxa, tibia, femur)
        if "raw_tune_cb" in self.callbacks:
            self.callbacks["raw_tune_cb"](msg[0:3])

    def setup_transition(self, target_s, target_t, target_k, duration=1.0):
        self.transitioning = True
        self.transition_elapsed = 0.0
        self.transition_duration = duration
        self.transition_target = (target_s, target_t, target_k)
        
        self.transition_initial = {
            leg: (val["shoulder"], val["thigh"], val["leg"]) 
            for leg, val in self.phi.items()
        }

    def process_transition(self, t):
        if not self.transitioning:
            return

        fraction = min(t / self.transition_duration, 1.0)
        
        target_s, target_t, target_k = self.transition_target
        current_angle = []

        for leg_key, leg_dict in self.phi.items():
            init_s, init_t, init_k = self.transition_initial[leg_key]

            leg_dict["shoulder"] = init_s + fraction * (target_s - init_s)
            leg_dict["thigh"] = init_t + fraction * (target_t - init_t)
            leg_dict["leg"] = init_k + fraction * (target_k - init_k)

            current_angle.extend([leg_dict["shoulder"], leg_dict["thigh"], leg_dict["leg"]])
        
        if "transition_cb" in self.callbacks:
            self.callbacks["transition_cb"](current_angle)

        if fraction >= 1.0:
            self.transitioning = False
            self.state_start_time = self.sim_time

    def gait_process(self, t):
        # NOTE: WALK | CRAWL | RUN | TURN uses this process
        planner = self.planner
        assert isinstance(planner, GaitPlanner)
        p = planner.p

        # NOTE: 1 second startup transition blend
        ramp = min(t / 1.0, 1.0)
        planner.stride_scale = ramp
        self.x_off_used = self.start_x_off + ramp * (p.x_off - self.start_x_off)
        self.z_off_used = self.start_z_off + ramp * (p.z_off - self.start_z_off)

        # NOTE: on turning
        if self.turning:
            if abs(self.yaw_rate) < 0.1:
                self.yaw_rate = p.sc_yaw
            planner.yaw_rate = self.yaw_rate

        # NOTE: Joint-space targets only for a PD consumer (ROS); the MPC uses the planner schedule
        if "walk_points" not in self.callbacks:
            return

        q_desired, q_dot_desired = [], []
        for leg in LEG_NAMES:
            ix, iy, iz = self.kinematics.get_init_pos(leg) # Get foot initial pos
            dx, dy, dz = planner.foot_offset(leg, t) # get foot offset on planar plane

            tx = ix + self.x_off_used + dx
            ty = iy + dy
            tz = -self.z_off_used + dz
            theta1, theta2, theta3 = self.kinematics.ik(leg, tx, ty, tz)
            q_desired.extend([theta1, theta2, theta3])

            # NOTE: planner foot velocity into joint velocity (damped J^-1)
            j = self.kinematics.get_jacobian(leg, theta1, theta2, theta3)
            j_inv = j.T @ np.linalg.inv(j @ j.T + 1e-6 * np.eye(3))
            q_dot = np.clip(j_inv @ np.array(planner.foot_vel(leg, t)), -40.0, 40.0)
            q_dot_desired.extend(q_dot.tolist())

            # NOTE: cache the angle, transitions start from here
            self.phi[leg]["shoulder"] = theta1
            self.phi[leg]["thigh"] = theta2
            self.phi[leg]["leg"] = theta3

        if "graph" in self.callbacks:
            self.callbacks["graph"]([float(t), float(q_desired[1]), float(self.current_q[1])])
        # NOTE: The MPC gets node 0 of the same schedule
        node0 = planner.horizon(t, 1, self.dt, self.robot_mass)
        points = [{
            "positions": q_desired,
            "velocities": q_dot_desired,
            "foot_forces": node0["fc_guess"][0],
            "is_stance": node0["stance"][0],
            "time_offset": 0.0
        }]

        self.callbacks["walk_points"](points)

    def jump_process(self, t):
        jp = self.planner
        assert isinstance(jp, JumpPlanner)
        name = jp.jump_phase(t)

        # NOTE: Event 1, PREPARE -> THRUST once prepare_time has passed
        # and the crouch height is actually reached
        if name == "PREPARE" and t >= jp.p.prepare_time:
            fk = self.kinematics.fk('FL', m.degrees(self.current_q[0]), m.degrees(self.current_q[1]), m.degrees(self.current_q[2]))
            if abs(abs(fk[1, 3]) - jp.p.y_crouch) < 0.1:
                jp.start_phase("THRUST", t)
                name = "THRUST"

        # NOTE: Event 2, early touchdown during DESCENT jumps to LANDING
        if name == "DESCENT" and (t - jp.starts["DESCENT"]) > 0.05 and np.mean(self.filtered_fz) > 2.0:
            jp.start_phase("LANDING", t)
            name = "LANDING"

        if jp.finished(t):
            self.update_state("IDLE")
            return

        self.jump_state = name

        # NOTE: IK at t - dt, t, t + dt with the phase locked, so the
        # central difference never mixes two phases' curves
        q_eval = []
        for t_eval in (t - self.dt, t, t + self.dt):
            x, y = jp.posture(name, t_eval)
            q = []
            for leg in LEG_NAMES:
                ix, iy, iz = self.kinematics.get_init_pos(leg)
                q.extend(self.kinematics.ik(leg, ix + x, y, iz))
            q_eval.append(np.array(q))

        q_d = q_eval[1]
        qd_d = (q_eval[2] - q_eval[0]) / (2.0 * self.dt)

        if "graph" in self.callbacks:
            self.callbacks["graph"]([float(t), float(q_d[1]), float(self.current_q[1])])

        # NOTE: node 0 of the same schedule the MPC gets
        node0 = jp.horizon(t, 1, self.dt, self.robot_mass)
        if "jump_points" in self.callbacks:
            self.callbacks["jump_points"](q_d.tolist(), qd_d.tolist(), node0["fc_guess"][0], node0["stance"][0])

    def loop_step(self, sim_time):
        self.sim_time = sim_time
        self._apply_pending(sim_time)
        if self.state_start_time is None:
            self.state_start_time = sim_time

        t = sim_time - self.state_start_time
        self.t = t

        if self.transitioning:
            self.process_transition(t)
        elif self.walking or self.turning:
            self.gait_process(t)
        elif self.jump_state != "":
            self.jump_process(t)

    def mpc_horizon(self, sim_time, n, dt):
        if self.transitioning or self.planner is None or self.state_start_time is None:
            return self.stand_planner.horizon(0.0, n, dt, self.robot_mass)
        return self.planner.horizon(sim_time - self.state_start_time, n, dt, self.robot_mass)

    def wbc_targets(self, sim_time):
        # NOTE: Single-instant schedule for the WBC; anything that is not a periodic gait
        # (transitions, jump) stands on all four feet for now
        if self.transitioning or self.state_start_time is None or not isinstance(self.planner, GaitPlanner):
            return self.stand_planner.instant(0.0, self.robot_mass)
        return self.planner.instant(sim_time - self.state_start_time, self.robot_mass)

