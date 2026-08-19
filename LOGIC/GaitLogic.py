import time
import math as m
import numpy as np
from LOGIC.KinematicsLogic import KinematicsLogic

# NOTE: ---------------- CONSTANT VALUE ----------------
LEG_NAMES = [
    'FL',
    'FR',
    'BL',
    'BR'
]

JOINT_NAMES = {
    'FL': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
    'FR': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
    'BL': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
    'BR': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint'],
}

LEG_TO_PHI = {
    'FL': 'tl_leg',
    'FR': 'tr_leg',
    'BL': 'bl_leg',
    'BR': 'br_leg'
}

class GaitLogic():
    def __init__(self, callbacks=None):
        self.kinematics = KinematicsLogic()
        self.callbacks = callbacks if callbacks else {}

        self.control_rate = 50.0 
        self.dt = 1.0 / self.control_rate # Time step (0.02s)
        self.t = 0.0 # global timer (s) incremented everey walk_process cycle (0 on enter)
        self.graph_t = 0.0
        self.robot_mass = 12.2

        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0

        # NOTE: Inverse Dynamics Parameters (Velocity & Acceleration)
        self.current_q = np.zeros(12)
        self.current_q_dot = np.zeros(12)
        self.filtered_fz = np.zeros(4)

        # NOTE: Every leg part index / tag
        self.phi = {
            "tr_leg": {
                "shoulder": 0.0,
                "thigh": 0.0,
                "leg": 0.0,
            },
            "tl_leg": {
                "shoulder": 0.0,
                "thigh": 0.0,
                "leg": 0.0,
            },       
            "br_leg": {
                "shoulder": 0.0,
                "thigh": 0.0,
                "leg": 0.0,
            },
            "bl_leg": {
                "shoulder": 0.0,
                "thigh": 0.0,
                "leg": 0.0,
            },       
        }

        self.walking = False
        self.jump_state = ""
        self.jump_q_history = np.zeros(12)
        self.jump_qd_history = np.zeros(12)

        self.transitioning = False
        self.transition_start_time = 0.0
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
    def update_state(self, msg: str):
        # NOTE: if state_callback is triggered again, 
        # it will cancel the walk timer, to avoid duplicated process
        self.walking = False
        self.jump_state = ""
        self.transitioning = False
        self.current_state = msg

        match msg:
            case "TUNING": 
                # NOTE: handled on GUI
                pass
            case "CROUCH": 
                # NOTE: just sends a target to lerp into
                self.setup_transition(0.00, 1.30, -2.70)
            case "IDLE": 
                # NOTE: just sends a target to lerp into
                self.setup_transition(0.00, 0.45, -0.90)
            case "WALK": 
                # NOTE: create a timed process for a walk process

                self.gait_freq = 1.0
                self.x_off = 0.25
                self.z_off = 2.5
                self.step_len = 1.5
                self.step_h = 1.0
                self.sc_yaw = 0.9

                self.t = 0.0
                self.walking = True
                self.duty_factor = 0.5
                self.z_off_used = self.z_off 
                self.target_yaw = self.current_yaw
                self.phase_offsets = {
                    'FL': 0.0, # (0 / 360 degree)
                    'BR': 0.0, # (90 degree)
                    'FR': m.pi, # (180 degree)
                    'BL': m.pi, # (270 degree)
                }
            case "JUMP":
                # NOTE: create a timed process for the jump process
                self.setup_transition(0.00, 0.45, -0.90)
                self.t = 0.0
                self.jump_state = "PREPARE"
                self.graph_t = 0.0
                self.jump_q_history = np.copy(self.current_q)
                self.jump_qd_history = np.zeros(12)

            case "CRAWL":
                self.gait_freq = 2.0
                self.x_off = 0.35
                self.z_off = 1.8
                self.step_len = 0.8
                self.step_h = 0.5
                self.sc_yaw = 0.9

                self.t = 0.0
                self.walking = True
                self.z_off_used = self.z_off 
                self.target_yaw = self.current_yaw
            case "RUN":
                
                self.gait_freq = 2.75
                self.x_off = 0.75
                self.z_off = 2.5
                self.step_len = 3.0
                self.step_h = 2.0
                self.sc_yaw = 0.9

                self.t = 0.0
                self.walking = True
                self.z_off_used = self.z_off
                self.target_yaw = self.current_yaw
                self.duty_factor = 0.50

                self.phase_offsets = {
                    'FL': 0.0, # (0 / 360 degree)
                    'BR': m.pi, # (90 degree)
                    'FR': 0.0, # (180 degree)
                    'BL': m.pi, # (270 degree)
                }

    def update_wt_params(self, msg: list):
        # NOTE: just sets the Walk Tune Param into a new Value from msg
        self.gait_freq = msg[0]
        self.x_off = msg[1]
        self.z_off = msg[2]
        self.step_len = msg[3]
        self.step_h = msg[4]
        self.sc_yaw = msg[5]

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
        print("wei")

    def update_phase_offsets(self, msg: list):
        # NOTE: just sets the Phase Offsets into a new Value from msg
        self.phase_offsets = {
            'FL': msg[0],
            'BR': msg[1],
            'FR': msg[2],
            'BL': msg[3],
        }

    def update_gamepad_params(self, msg: list):
        if not self.walking:
            return

        ly = msg[0]
        rx = msg[1]
        
        forward_dir = 1.0 if ly < 0 else -1.0
        scale = 3.0 if self.current_state == "RUN" else 1.0
        self.step_len = forward_dir * scale 

        yaw_turn_rate = 0.02
        self.target_yaw += rx * yaw_turn_rate

    def raw_tune(self, msg: list):
        # NOTE: Raw Tuning just sends a theta value from msg (per coxa, tibia, femur)
        self.send_theta(msg[0], msg[1], msg[2])

    def trajectory_controller(self, phase, step_len):
        z = 0.0
        # NOTE: z value will always be 0.0, (depth if the legs is viewed as a 2d leg)

        stance_phase_end = 2.0 * m.pi * self.duty_factor
        swing_phase_len = 2.0 * m.pi * (1.0 - self.duty_factor)

        # print(f"duty fact : {self.duty_factor}, stance len : {stance_phase_end}, swing phase : {swing_phase_len} ")

        if phase < stance_phase_end:
            # NOTE: stance phase : 
            # foot is on the ground moving backward relative to body
            # - fraction goes from 0 -> 1
            # - x goes from -step_len/2 to step_len/2 ([-] is somehow forward)
            # - y = 0 we want the leg to stay on ground

            fraction = phase / stance_phase_end 
            x = -(step_len / 2.0) + (fraction * step_len)
            y = 0.0
        else: # NOTE: swing phase : foot is off ground arc-ing forward
            # - fraction goes from 0 -> 1
            # - x goes from +step_len/2 to -step_len/2
            # - y follows a half-sine arc : 0 -> step_h -> 0

            s = (phase - stance_phase_end) / swing_phase_len
            v = step_len * (swing_phase_len / stance_phase_end)
            c = step_len + v
            x = (step_len / 2.0) + (v * s) - (10.0 * c * s**3) + (15.0 * c * s**4) - (6.0 * c * s**5)
            y = 64.0 * self.step_h * (s**3) * ((1.0 - s)**3)

        return x, y, z

    def run_trajectory_controller(self, phase, step_len, leg):
        z = 0.0
        stance_phase_end = 2.0 * m.pi * self.duty_factor
        swing_phase_len = 2.0 * m.pi * (1.0 - self.duty_factor)

        # Asymmetric Stride: Front legs catch (reach forward), Back legs push (sweep back)
        if leg in ['BL', 'BR']:
            x_start = (step_len * 0.8)
            x_end = (step_len * 0.2)
        else:
            x_start = (step_len * 0.2)
            x_end = (step_len * 0.8)

        if phase < stance_phase_end:
            # Stance Phase: Sweep flat on the ground
            fraction = phase / stance_phase_end 
            x = x_start + fraction * (x_end - x_start)
            y = 0.0 
        else: 
            # Swing Phase: Return arc
            s = (phase - stance_phase_end) / swing_phase_len
            
            # Your exact inverted target path
            target_x = -x_end + (s * (x_end - x_start))
            
            # LERP the teleportation jump over the first 15% of the swing
            if s < 0.15:
                blend = s / 0.15
                x = (1.0 - blend) * x_end + blend * target_x
            else:
                x = target_x
                
            y = 64.0 * self.step_h * (s**3) * ((1.0 - s)**3)

        return x, y, z

    def setup_transition(self, target_s, target_t, target_k, duration=1.0):
        self.transitioning = True
        self.transition_start_time = time.time()
        self.transition_duration = duration
        self.transition_target = (target_s, target_t, target_k)
        
        self.transition_initial = {
            leg: (val["shoulder"], val["thigh"], val["leg"]) 
            for leg, val in self.phi.items()
        }

    def process_transition(self):
        if not self.transitioning:
            return

        elapsed = time.time() - self.transition_start_time
        fraction = min(elapsed / self.transition_duration, 1.0)

        target_s, target_t, target_k = self.transition_target
        current_angles = []

        for leg_key, leg_dict in self.phi.items():
            init_s, init_t, init_k = self.transition_initial[leg_key]
            
            leg_dict["shoulder"] = init_s + fraction * (target_s - init_s)
            leg_dict["thigh"] = init_t + fraction * (target_t - init_t)
            leg_dict["leg"] = init_k + fraction * (target_k - init_k)
            
            current_angles.extend([leg_dict["shoulder"], leg_dict["thigh"], leg_dict["leg"]])

        if "transition_cb" in self.callbacks:
            self.callbacks["transition_cb"](current_angles)

        if fraction >= 1.0:
            self.transitioning = False

    def walk_process(self):
        # NOTE: Angular Frequency
        # rate of change of a phase angle with respect to time in rotational
        # or a periodic motion, it's general equation are :
        # w = 2*pi*f = 2pi/T
        omega = 2.0 * m.pi * self.gait_freq

        # NOTE: sets the trajectory to look forward ahead (10 steps) 
        lookahead_steps = 10 
        points = []
        all_pos_history = []
        is_stance_history = []

        # NOTE: Create a linear ramp multiplier from 0.0 to 1.0 over the first 1s
        # to scale the base step length
        ramp_duration = 1.0
        ramp_factor = min(self.t / ramp_duration, 1.0)
        base_step_len = self.step_len * ramp_factor

        for i in range(lookahead_steps):
            # NOTE: Calculate target timestamp for the current lookahead_steps
            t_ahead = self.t + i * self.dt

            # NOTE: Get the Modulo of the phase angle
            # With the equation : (w * t_ahead) mod (2pi)
            # its a reference clock as in the omega is the parent. the phase_now
            # is what currently time it is kinda thing. it's mapping a linear time
            # into angular (doesnt mean shit for me) or into a cyclical value between
            # 0 and 2pi radians.
            phase_now = (omega * t_ahead) % (2.0 * m.pi)

            # NOTE: Initialize an empty lists for the target joints angles and stances
            # for this specific time step.
            q_desired = []
            is_stance_list = []

            # NOTE: get yaw error
            yaw_error = self.target_yaw - self.current_yaw

            stance_limit = 2.0 * m.pi * self.duty_factor

            for leg in LEG_NAMES:
                # NOTE: leg specific phase shift 
                # leg_phase is the most local shit of time after phase_now
                # calculate or timed each own leg cycle. calculated by adding
                # a static angular offset to the phase_now. 
                # it tells which leg should start doing shit first by 
                # adding the offset shit
                leg_phase = (phase_now + self.phase_offsets[leg]) % (2.0 * m.pi)
                is_stance = (leg_phase < stance_limit) # check if leg is in the stance phase
                is_stance_list.append(is_stance) # record and set the starting step

                active_step_len = base_step_len
                if i == 0 and leg == 'FL':
                    traj_stance = leg_phase < (2.0 * m.pi * self.duty_factor)
                    physics_stance = is_stance
                    if traj_stance != physics_stance:
                        print(f"DESYNC ERROR | Phase: {leg_phase:.2f} | Trajectory Stance: {traj_stance} | Physics Stance: {physics_stance}")

                # NOTE: Adjusts the step length asymmetrically based on yaw error 
                # to induce rotation. Left legs increase stride, 
                # right legs decrease stride (or vice versa).
                if leg in ['FL', 'BL']:
                    active_step_len += (yaw_error * self.sc_yaw) * ramp_factor
                elif leg in ['FR', 'BR']:
                    active_step_len -= (yaw_error * self.sc_yaw) * ramp_factor

                # NOTE: Calculate the cartesian foot trajectory offset for the 
                # current phase
                if self.current_state == "RUN":
                    xl, yl, zl = self.run_trajectory_controller(leg_phase, active_step_len, leg)
                else:
                    xl, yl, zl = self.trajectory_controller(leg_phase, self.step_len)

                # NOTE: Retrieves the static nominal resting position of the leg
                ix, iy, iz = self.kinematics.get_init_pos(leg)

                # NOTE: Calculate the absolute target foot coordinates
                tx = ix + xl + self.x_off
                ty = self.z_off_used - yl
                tz = iz + zl

                # NOTE: Calculate Inverse Kinematics to find Theta
                theta1, theta2, theta3 = self.kinematics.ik(leg, tx, ty, tz)
                q_desired.extend([theta1, theta2, theta3]) # appends theta for graph 
                
                # NOTE: cache the calculated angle into the leg dict
                if i == 0:
                    phi_key = LEG_TO_PHI[leg]
                    self.phi[phi_key]["shoulder"] = theta1
                    self.phi[phi_key]["thigh"] = theta2
                    self.phi[phi_key]["leg"] = theta3

            # NOTE: appends all position and stance to an array for tracking
            q_d = np.array(q_desired)
            all_pos_history.append(q_d)
            is_stance_history.append(is_stance_list)

        # NOTE: Calculate the velocities (q dot) with Angular Wrapping
        # This prevents the "infinity shit" when the joint wraps around 360 degrees
        all_vel_history = []
        for i in range(lookahead_steps):
            if i == 0:
                diff = all_pos_history[1] - all_pos_history[0]
                diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                q_dot_d = diff / self.dt
            elif i == lookahead_steps - 1:
                diff = all_pos_history[i] - all_pos_history[i - 1]
                diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                q_dot_d = diff / self.dt
            else:
                diff = all_pos_history[i + 1] - all_pos_history[i - 1]
                diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                q_dot_d = diff / (2.0 * self.dt)
            all_vel_history.append(q_dot_d)

        # NOTE: Calculate the acceleration and clamp it
        # Prevents finite-difference spikes from blowing up the physics solver mass matrix
        all_acc_history = []
        for i in range(lookahead_steps):
            if i == 0:
                diff = all_vel_history[1] - all_vel_history[0]
            elif i == lookahead_steps - 1:
                diff = all_vel_history[i] - all_vel_history[i - 1]
            else:
                diff = (all_vel_history[i + 1] - all_vel_history[i - 1]) / 2.0
            
            q_ddot_d = diff / self.dt
            q_ddot_d = np.clip(q_ddot_d, -1500.0, 1500.0) 
            all_acc_history.append(q_ddot_d)

        # NOTE: Initialize the torque matrix first (array really)
        torque = np.zeros(14) 

        # NOTE: Compute torques for each lookahead_steps
        for i in range(lookahead_steps):

            # NOTE: get all the previously Calculated velocities and accelerations
            q_d = all_pos_history[i]
            q_dot_d = all_vel_history[i]
            q_ddot_d = all_acc_history[i]
            is_stance_list = is_stance_history[i]

            # NOTE: for graph, track starts here
            if i == 0 and "graph" in self.callbacks:
                self.callbacks["graph"]([
                    float(self.t),
                    float(q_d[1]),
                    float(self.current_q[1])
                ])

            # NOTE: Initialize the foot force matrix and Counts the active stance legs
            foot_forces = np.zeros((4, 3))
            stance_count = sum(is_stance_list)

            # NOTE: Calculate the basic static weight distribution 
            # across active stance feet.
            # Calculate the basic static weight distribution 
            if stance_count > 0:
                base_weight_per_foot = (self.robot_mass * 9.81) / stance_count
                t_ahead = self.t + i * self.dt
                phase_now = (omega * t_ahead) % (2.0 * m.pi)
                stance_limit = 2.0 * m.pi * self.duty_factor

                for leg_idx, leg_name in enumerate(LEG_NAMES):
                    leg_phase = (phase_now + self.phase_offsets[leg_name]) % (2.0 * m.pi)
                    
                    # Apply force only during stance phase
                    if leg_phase < stance_limit:
                        stance_fraction = leg_phase / stance_limit
                        
                        # Sinusoidal multiplier peaks at mid-stance
                        dynamic_multiplier = 1.0 + m.sin(stance_fraction * m.pi)
                        
                        if self.current_state == "RUN":
                            dynamic_multiplier *= 1.75 # Extra peak force to achieve flight phase

                        foot_forces[leg_idx, 2] = base_weight_per_foot * dynamic_multiplier

            points.append({
                "positions": q_d.tolist(),
                "velocities": q_dot_d.tolist(),
                "accelerations": q_ddot_d.tolist(),
                "foot_forces": foot_forces.tolist(),
                "is_stance": is_stance_list,
                "time_offset": i * self.dt
            })

        if "walk_points" in self.callbacks:
            self.callbacks["walk_points"](points)

        self.t += self.dt # increment the time mod (t)

    def jump_process(self):
        y_idle = self.z_off
        y_crouch = self.y_crouch
        y_thrust = self.y_thrust
        y_flight = self.y_flight

        x_idle = self.x_off
        x_thrust = self.x_off + self.back_thrust 
        x_stabilize = self.x_off + self.x_stabilize
        x_rev_thrust = self.x_off - self.x_thrust
        x_flight = self.x_off + self.x_flight
        x_catch = self.x_off + self.x_catch

        current_x = x_idle
        current_y = y_idle

        avg_fz = np.mean(self.filtered_fz)
        contact_threshold = 2.0
        
        foot_forces = np.zeros((4, 3))
        is_stance = [False, False, False, False]
        base_weight = (self.robot_mass * 9.81) / 4.0

        t1 = self.prepare_time
        t2 = t1 + self.back_thrust_time
        t3 = t2 + self.flight_time
        t4 = t3 + self.catch_time

        # Fast-forward override
        if self.jump_state == "DESCENT" and (self.t - t3) > 0.05 and avg_fz > contact_threshold:
            self.t = t4
            self.jump_state = "LANDING"

        # 1. Lock active state to prevent boundary whiplash in Central Difference
        active_state = self.jump_state
        t_start = 0.0
        duration = 1.0

        if active_state == "PREPARE":
            t_start = 0.0
            duration = self.prepare_time
        elif active_state == "THRUST":
            t_start = t1
            duration = self.back_thrust_time
        elif active_state == "FLIGHT":
            t_start = t2
            duration = self.flight_time
        elif active_state == "DESCENT":
            t_start = t3
            duration = self.catch_time
        elif active_state == "LANDING":
            t_start = t4
            duration = self.landing_time

        q_eval = []
        current_y_actual = 0.0

        for step in [-1, 0, 1]:
            t_eval = self.t + step * self.dt
            fraction = (t_eval - t_start) / duration 
            # DO NOT CLIP FRACTION. Allow math to extrapolate tangents naturally.
            
            match active_state:
                case "PREPARE":
                    s = 3.0 * (fraction**2) - 2.0 * (fraction**3) 
                    current_y = y_idle + s * (y_crouch - y_idle) 
                    current_x = x_idle + s * (x_stabilize - x_idle) 
                    if step == 0:
                        is_stance = [True, True, True, True]
                        for i in range(4): foot_forces[i, 2] = base_weight
                        t_fk1, t_fk2, t_fk3 = m.degrees(self.current_q[0]), m.degrees(self.current_q[1]), m.degrees(self.current_q[2])
                        fk_matrix = self.kinematics.fk('FL', t_fk1, t_fk2, t_fk3) 
                        current_y_actual = abs(fk_matrix[1, 3])

                case "THRUST":
                    s = fraction ** 2
                    current_y = y_crouch + s * (y_thrust - y_crouch)
                    current_x = x_stabilize + s * (x_thrust - x_stabilize)
                    if step == 0:
                        is_stance = [True, True, True, True]
                        for i in range(4): foot_forces[i, 2] = base_weight * 2.5 
                        
                case "FLIGHT":
                    s = fraction 
                    current_y = y_thrust + s * (y_flight - y_thrust)
                    current_x = x_thrust + s * (x_flight - x_thrust)
                    if step == 0: is_stance = [False, False, False, False]

                case "DESCENT":
                    s = 3.0 * (fraction**2) - 2.0 * (fraction**3)
                    current_y = y_flight 
                    current_x = x_flight + s * (x_catch - x_flight)
                    if step == 0: is_stance = [False, False, False, False]

                case "LANDING":
                    s = 3.0 * (fraction**2) - 2.0 * (fraction**3)
                    current_y = y_flight + s * (y_idle - y_flight)
                    current_x = x_catch + s * (x_idle - x_catch)
                    if step == 0:
                        is_stance = [True, True, True, True]
                        for i in range(4): foot_forces[i, 2] = base_weight * 1.5

            q_desired = []
            for leg in LEG_NAMES:
                ix, iy, iz = self.kinematics.get_init_pos(leg)
                tx, ty, tz = ix + current_x, current_y, iz 
                theta1, theta2, theta3 = self.kinematics.ik(leg, tx, ty, tz)
                q_desired.extend([theta1, theta2, theta3])
                
            q_eval.append(np.array(q_desired))

        # Predictive Central Difference Array Matrix 
        q_d = q_eval[1]
        qd_d = (q_eval[2] - q_eval[0]) / (2.0 * self.dt)
        qdd_d = (q_eval[2] - 2.0 * q_eval[1] + q_eval[0]) / (self.dt**2)
        qdd_d = np.clip(qdd_d, -1500.0, 1500.0)

        if self.jump_state == "THRUST":
            print(f"[DEBUG] t: {self.t:.3f} | q_d (pos): {q_d[1]:.3f} | qd_d (vel): {qd_d[1]:.3f} | qdd_d (accel): {qdd_d[1]:.3f}")

        # Handle state transitions based on absolute time
        match self.jump_state:
            case "PREPARE":
                if self.t >= t1 and abs(current_y_actual - y_crouch) < 0.1:
                    self.jump_state = "THRUST"
            case "THRUST":
                if self.t >= t2:
                    self.jump_state = "FLIGHT"
            case "FLIGHT":
                if self.t >= t3:
                    self.jump_state = "DESCENT"
            case "DESCENT":
                # Descent fast-forward is handled at top of loop
                if self.t >= t4:
                    self.jump_state = "LANDING"
            case "LANDING":
                if self.t >= (t4 + self.landing_time):
                    self.update_state("IDLE")
                    return

        if "graph" in self.callbacks:
            self.callbacks["graph"]([
                float(self.graph_t),
                float(q_d[1]),
                float(self.current_q[1])
            ])

        if "jump_points" in self.callbacks:
            self.callbacks["jump_points"](q_d.tolist(), qd_d.tolist(), qdd_d.tolist(), foot_forces.tolist(), is_stance)

        self.t += self.dt
        self.graph_t += self.dt

    def loop_step(self):
        if self.transitioning:
            self.process_transition()
        elif self.walking:
            self.walk_process()
        elif self.jump_state != "":
            self.jump_process()
