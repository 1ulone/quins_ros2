import enum
import os
import time
import rclpy

import math as m 
import numpy as np
import pinocchio as pin

from rclpy.node import Node
from typing import Optional
from nav_msgs.msg import Odometry 
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from LOGIC.KinematicsLogic import KinematicsLogic
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Float64MultiArray, Int8MultiArray
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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

# NOTE: ---------------- MAIN CLASS TUNER ----------------
class Tuner(Node):
    # NOTE: -------- Initialization --------
    def __init__(self):
        super().__init__('quins_tuner')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', Parameter.Type.BOOL, True)]) # Force rclpy to use Simulation Time 
        self.kinematics = KinematicsLogic() # KinematicsLogic Reference

        # NOTE: Publisher 
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.graph_pub = self.create_publisher(Float64MultiArray, '/tuner/graph', 10)
        self.contact_pub = self.create_publisher(Int8MultiArray, 'tuner/contacts', 10)

        # NOTE: Subcriber
        self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        self.create_subscription(String, '/tuner/state', self.state_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/raw', self.raw_tune_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/params', self.params_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/jparams', self.jump_params_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/phase_offsets', self.phase_callback, 10)

        # NOTE: Timer
        self.control_rate = 50.0 # Control loop freq (2ms)
        self.dt = 1.0 / self.control_rate # Time step (0.02s)
        self.t = 0.0 # global timer (s) incremented everey walk_process cycle (0 on enter)

        # NOTE: Model private variable
        package_path = get_package_share_directory('quins')
        urdf_path = os.path.join(package_path, 'urdf', 'quadruped.urdf')
        self.pin_model = pin.buildModelsFromUrdf(urdf_path)[0]
        self.pin_data = self.pin_model.createData()
        self.robot_mass = 10.0

        # NOTE: PID private variable 
        self.current_pos = np.zeros(3)
        self.current_lin_vel = np.zeros(3)
        self.current_ang_vel = np.zeros(3)
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.target_yaw = 0.0

        # NOTE: Walk Private Variable
        self.walking = False
        self.walk_start_time = None
        self.walk_timer: Optional[rclpy.timer.Timer] = None
        self.last_cycle_wall_time = None
        self.last_join_state_stamp = None

        # NOTE: Jump Private Variable
        self.jump_state = ""
        self.jump_timer: Optional[rclpy.timer.Timer] = None

        # NOTE: Walk Tune Parameters
        self.gait_freq = 1.0
        self.x_off = 0.25
        self.z_off = 2.7
        self.step_len = 2.0
        self.step_h = 0.75
        self.sc_yaw = 0.9

        # NOTE: Jump Tune Parameters
        self.y_crouch = 1.0
        self.y_thrust = 3.8
        self.y_flight = 1.8
        self.x_thrust = 0.5
        self.x_flight =-1.5
        self.x_catch = -1.5
        self.prepare_time = 0.8
        self.thrust_time = 0.15
        self.flight_time = 0.05
        self.landing_time = 0.5
        self.catch_time = 0.1
        self.graph_t = 0.0

        # NOTE: Inverse Dynamics Parameters (Velocity & Acceleration)
        self.current_q = np.zeros(12)
        self.current_q_dot = np.zeros(12)
        self.filtered_fz = np.zeros(4)

        # NOTE: Phase Offsets
        # 3.14 -> 360 degree, radian to angle 
        # phase offsets results in a full radian cycle 0 - 3.14
        self.phase_offsets = {
            'FL': 0.0, # (0 / 360 degree)
            'BR': m.pi / 2.0, # (90 degree)
            'FR': m.pi, # (180 degree)
            'BL': 3.0 * m.pi / 2.0, # (270 degree)
        }

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



    # NOTE: -------- Callback --------
    def state_callback(self, msg: String):
        # NOTE: if state_callback is triggered again, 
        # it will cancel the walk timer, to avoid duplicated process
        self.walking = False
        if hasattr(self, 'walk_timer') and self.walk_timer is not None:
            self.walk_timer.cancel()
            self.walk_timer = None

        if hasattr(self, 'jump_timer') and self.jump_timer is not None:
            self.jump_timer.cancel()
            self.jump_timer = None

        match msg.data:
            case "TUNING": 
                # NOTE: handled on GUI
                pass
            case "CROUCH": 
                # NOTE: just sends a target to lerp into
                self.animate_transition(0.00, 1.30, -2.70)
            case "IDLE": 
                # NOTE: just sends a target to lerp into
                self.animate_transition(0.00, 0.45, -0.90)
            case "WALK": 
                # NOTE: create a timed process for a walk process
                self.target_yaw = self.current_yaw
                self.t = 0.0
                self.walk_start_time = self.get_clock().now()
                self.walking = True

                self.walk_timer = self.create_timer(
                    self.dt,
                    lambda: self.walk_process(self.kinematics)
                )
            case "JUMP":
                # NOTE: create a timed process for the jump process
                self.t = 0.0
                self.jump_state = "PREPARE"
                self.graph_t = 0.0
                self.jump_timer = self.create_timer(
                    self.dt,
                    lambda:self.jump_process(self.kinematics)
                )

    def params_callback(self, msg: Float64MultiArray):
        # NOTE: just sets the Walk Tune Param into a new Value from msg
        self.gait_freq = msg.data[0]
        self.x_off = msg.data[1]
        self.z_off = msg.data[2]
        self.step_len = msg.data[3]
        self.step_h = msg.data[4]
        self.sc_yaw = msg.data[5]

    def jump_params_callback(self, msg: Float64MultiArray):
        # NOTE: just sets the Jump Tune Param into a new Value from msg
        self.y_crouch = msg.data[0]
        self.y_thrust = msg.data[1] 
        self.y_flight = msg.data[2] 
        self.x_thrust = msg.data[3]
        self.x_flight = msg.data[4]
        self.x_catch = msg.data[5]
        self.prepare_time = msg.data[6] 
        self.thrust_time = msg.data[7] 
        self.flight_time = msg.data[8]
        self.landing_time = msg.data[9]
        self.catch_time = msg.data[10]

    def phase_callback(self, msg: Float64MultiArray):
        # NOTE: just sets the Phase Offsets into a new Value from msg
        self.phase_offsets = {
            'FL': msg.data[0],
            'BR': msg.data[1],
            'FR': msg.data[2],
            'BL': msg.data[3],
        }

    def raw_tune_callback(self, msg: Float64MultiArray):
        # NOTE: Raw Tuning just sends a theta value from msg (per coxa, tibia, femur)
        self.send_theta(msg.data[0], msg.data[1], msg.data[2])

    def joint_state_callback(self, msg: JointState):
        expected_order = [
            'tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint',
            'tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint',
            'bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint',
            'br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint',
        ]

        q_sorted = np.zeros(12)
        q_dot_sorted = np.zeros(12)
        tau_sorted = np.zeros(12)

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, joint_name in enumerate(expected_order):
            if joint_name in name_to_idx:
                idx = name_to_idx[joint_name]
                q_sorted[i] = msg.position[idx]
                q_dot_sorted[i] = msg.velocity[idx]
                tau_sorted[i] = msg.effort[idx]

        self.current_q = q_sorted
        self.current_q_dot = q_dot_sorted

        contact_states, vertical_forces = self.estimate_grf_contact(
            self.current_q,
            self.current_q_dot,
            tau_sorted
        )

        contacts = Int8MultiArray()
        contacts.data = contact_states
        self.contact_pub.publish(contacts)

        self.last_join_state_stamp = self.get_clock().now()

    def odometry_callback(self, msg: Odometry):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        self.current_lin_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

        q = msg.pose.pose.orientation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z

        # NOTE: 
        #  Inputing Quaternion q = (x, y, z) from odom message
        #  then Outputing Yaw Angle in Radians
        #  using the equation :
        #  sin_y = 2(w*z + x*y)
        #  cos_y = 1 - 2(y^2 + z^2)
        #  yaw = atan2(sin_y, cos_y)

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.current_roll = m.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        self.current_pitch = m.asin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

        self.current_yaw = m.atan2(siny_cosp, cosy_cosp)

        self.current_ang_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    def send_theta(self, coxa, femur, tibia):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = []
        
        point = JointTrajectoryPoint()
        point.positions = []

        # NOTE: iterate over all 4 legs in JOINT_NAMES order
        # appends the 3 joints names and the 3 angle values
        # results in 12 joint names and 12 positions
        # all legs identical angles
        for joints in JOINT_NAMES.values():
            msg.joint_names.extend(joints)
            point.positions.extend([coxa, femur, tibia])

        # NOTE: the controller will interpolate to the target position over 0.5s
        point.time_from_start = Duration(sec=0, nanosec=500000000)
        msg.points.append(point)  # type: ignore
        self.joint_pub.publish(msg)

    def animate_transition(self, target_s, target_t, target_k, duration=1.0):
        # NOTE: cancels any existing transition process
        if hasattr(self, 'transition_timer') and self.transition_timer is not None:
            self.transition_timer.cancel()
            self.transition_timer = None

        # NOTE: Records current start time, and stores initial joint
        start_time = time.time()
        initial_s = self.phi["tr_leg"]["shoulder"]
        initial_t = self.phi["tr_leg"]["thigh"]
        initial_k = self.phi["tr_leg"]["leg"]

        # NOTE: Lerping here
        def step():
            elapsed = time.time() - start_time
            fraction = min(elapsed / duration, 1.0)

            current_s = initial_s + fraction * (target_s - initial_s)
            current_t = initial_t + fraction * (target_t - initial_t)
            current_k = initial_k + fraction * (target_k - initial_k)

            for leg in self.phi.values():
                leg["shoulder"] = current_s
                leg["thigh"] = current_t 
                leg["leg"] = current_k 

            self.send_theta(current_s, current_t, current_k)

            if fraction >= 1.0:
                if hasattr(self, 'transition_timer') and self.transition_timer is not None:
                    self.transition_timer.cancel()
                    self.transition_timer = None

        # NOTE: loop def step while on transition process
        self.transition_timer = self.create_timer(0.02, step)

    def trajectory_controller(self, phase, step_len):
        z = 0.0
        # NOTE: z value will always be 0.0, (depth if the legs is viewed as a 2d leg)

        if phase < m.pi:
            # NOTE: stance phase : 
            # foot is on the ground moving backward relative to body
            # - fraction goes from 0 -> 1
            # - x goes from -step_len/2 to step_len/2 ([-] is somehow forward)
            # - y = 0 we want the leg to stay on ground

            fraction = phase / m.pi
            x = -(self.step_len / 2.0) + (fraction * step_len)
            y = 0.0
        else: # NOTE: swing phase : foot is off ground arc-ing forward
            # - fraction goes from 0 -> 1
            # - x goes from +step_len/2 to -step_len/2
            # - y follows a half-sine arc : 0 -> step_h -> 0

            fraction = (phase - m.pi) / m.pi
            x = (self.step_len / 2.0) - (fraction * step_len)
            y = self.step_h * m.sin(fraction * m.pi)

        return x, y, z

    # NOTE: Check if foot touching the ground and get ground reaction forces
    def estimate_grf_contact(self, q, q_dot, torque_measured, force_threshold=1.0):
        torque_expected = pin.rnea(self.pin_model, self.pin_data, q, q_dot, np.zeros_like(q_dot))
        torque_residual = torque_expected - torque_measured

        foot_frame_names = ['tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link']

        contact_state = []
        vertical_forces = []

        alpha = 0.15

        for i, frame_name in enumerate(foot_frame_names):
            frame_id = self.pin_model.getFrameId(frame_name)

            j_full = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            j_linear = j_full[:3, :]
            j_t_pinv = np.linalg.pinv(j_linear.T)
            f_ext = j_t_pinv @ torque_residual

            fz = f_ext[2]
            self.filtered_fz[i] = (alpha * fz) + ((1.0 - alpha) * self.filtered_fz[i])
            is_touching_ground = bool(self.filtered_fz[i] > force_threshold)

            contact_state.append(is_touching_ground)
            vertical_forces.append(fz)

        return contact_state, vertical_forces 

    # NOTE: reason as to why inverse dynamics is not inside KinematicsLogic.py
    # is because only Gazebo / Ros uses Pinocchio. So not really modular-able
    def inverse_dynamics(self, q, q_dot, q_ddot_cmd, foot_forces, is_stance_array):
        # NOTE: using Composite Rigid Body Algorithm (CRBA) to Compute Joint-space 
        # inertia matrix, see Documentation :
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html#af803e1af2a3be938bbfa5e270a02c4cd
        m_matrix = pin.crba(self.pin_model, self.pin_data, q)

        # NOTE: using Recursive Newton-Euler Algorithm (RNEA) to Compute Coriolis,
        # Centrifugal, and Gravity Forces by evaluating Inverse Dynamics with 
        # zero acceleration, see Documentation :
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html#ab48efbd527d1bc9941da1a5f400e751a
        bias_forces = pin.rnea(self.pin_model, self.pin_data, q, q_dot, np.zeros_like(q_dot))

        # NOTE: using the general Rigidbody Equation of Motion
        # Which is T = M(q)qddot + h(q,qdot) 
        torque = (m_matrix @ q_ddot_cmd) + bias_forces

        # NOTE: define four end foot effector
        foot_frame_names = ['bl_tip_link', 'br_tip_link', 'tl_tip_link', 'tr_tip_link']

        # NOTE: check if the foot is touching the ground (Check if on ground stance)
        # computeFrameJacobian then maps the joint Velocities into a 6D Cartesian
        # Velocities of the end effector
        for i, frame_name in enumerate(foot_frame_names):
            if is_stance_array[i]:
                frame_id = self.pin_model.getFrameId(frame_name)
                J_full = pin.computeFrameJacobian(
                    self.pin_model, 
                    self.pin_data, 
                    q, 
                    frame_id, 
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                
                # NOTE: extract only the top 3 rows of the jacobian matrix
                # retaining only the linear translation components (x, y, z)
                # discarding the angular components
                J_linear = J_full[:3, :]

                # NOTE: Translate Cartesian contact forces into Equivalent joint Torques
                torque -= (J_linear.T @ foot_forces[i])

        return torque 

    def walk_process(self, k: KinematicsLogic):
        # NOTE: if walk is already starting, then we sync time mod (t)
        # with current process else we reset time mod (t) and make 
        # walk_start_time to current time
        if self.walk_start_time is not None:
            self.t = (self.get_clock().now() - self.walk_start_time).nanoseconds * 1e-9
        else:
            self.walk_start_time = self.get_clock().now()
            self.t = 0.0

        # NOTE: Angular Frequency
        # rate of change of a phase angle with respect to time in rotational
        # or a periodic motion, it's general equation are :
        # w = 2*pi*f = 2pi/T
        omega = 2.0 * m.pi * self.gait_freq

        # NOTE: init joint message for publishing 
        msg = JointTrajectory()
        msg.joint_names = []

        # NOTE: setting the names for the joint_names 
        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]

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

            for leg in LEG_NAMES:
                # NOTE: leg specific phase shift 
                # leg_phase is the most local shit of time after phase_now
                # calculate or timed each own leg cycle. calculated by adding
                # a static angular offset to the phase_now. 
                # it tells which leg should start doing shit first by 
                # adding the offset shit
                leg_phase = (phase_now + self.phase_offsets[leg]) % (2.0 * m.pi)
                is_stance = (leg_phase >= m.pi) # check if leg is in the stance phase
                is_stance_list.append(is_stance) # record and set the starting step
                active_step_len = base_step_len

                # NOTE: Adjusts the step length asymmetrically based on yaw error 
                # to induce rotation. Left legs increase stride, 
                # right legs decrease stride (or vice versa).
                if leg in ['FL', 'BL']:
                    active_step_len += (yaw_error * self.sc_yaw) * ramp_factor
                elif leg in ['FR', 'BR']:
                    active_step_len -= (yaw_error * self.sc_yaw) * ramp_factor

                # NOTE: Calculate the cartesian foot trajectory offset for the 
                # current phase
                xl, yl, zl = self.trajectory_controller(leg_phase, self.step_len)

                # NOTE: Retrieves the static nominal resting position of the leg
                ix, iy, iz = k.get_init_pos(leg)

                # NOTE: Calculate the absolute target foot coordinates
                tx = ix + xl + self.x_off
                ty = self.z_off - yl
                tz = iz + zl

                # NOTE: Calculate Inverse Kinematics to find Theta
                theta1, theta2, theta3 = k.ik(leg, tx, ty, tz)
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

        # NOTE: Calculate the velocities (q dot) based on the calculated position 
        # which uses the equation : qdot = q_i+1 - q_i-1 / 2*delta_t
        all_vel_history = []
        for i in range(lookahead_steps):
            if i == 0:
                q_dot_d = (all_pos_history[1] - all_pos_history[0]) / self.dt
            elif i == lookahead_steps - 1:
                q_dot_d = (all_pos_history[i] - all_pos_history[i - 1]) / self.dt
            else:
                q_dot_d = (all_pos_history[i + 1] - all_pos_history[i - 1]) / (2.0 * self.dt)
            all_vel_history.append(q_dot_d)

        # NOTE: Calculate the acceleration (q double dot) based on the calculated velocities
        # which uses the equation : qddot = qdot_i+1 - qdot_i-1 / 2*delta_t
        all_acc_history = []
        for i in range(lookahead_steps):
            if i == 0:
                q_ddot_d = (all_vel_history[1] - all_vel_history[0]) / self.dt
            elif i == lookahead_steps - 1:
                q_ddot_d = (all_vel_history[i] - all_vel_history[i - 1]) / self.dt
            else:
                q_ddot_d = (all_vel_history[i + 1] - all_vel_history[i - 1]) / (2.0 * self.dt)
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

            if i == 0:
                # NOTE: for graph, track starts here
                graph_msg = Float64MultiArray()
                graph_msg.data = [
                    float(self.t),
                    float(q_d[1]),
                    float(self.current_q[1])
                ]
                self.graph_pub.publish(graph_msg)

                # NOTE: For debugging purpose
                # now = self.get_clock().now()
                # staleness = 0.0
                # if self.last_cycle_wall_time is not None:
                #     actual_dt = (now - self.last_cycle_wall_time).nanoseconds * 1e-9
                #     if abs(actual_dt - self.dt) > 1.5 * self.dt:
                #         self.get_logger().warn(
                #             f"its jittering: expected {self.dt*1000:.1f} ms, "
                #             f"got {actual_dt*1000:.1f} ms"
                #         )
                # self.last_cycle_wall_time = now
                #
                # if self.last_join_state_stamp is not None:
                #     staleness = (now - self.last_join_state_stamp).nanoseconds * 1e-9
                #     if staleness > 2.5 * self.dt:
                #         self.get_logger().warn(f"current_q is {staleness*1000:.1f} ms stale at consumption")

            # NOTE: Initialize the foot force matrix and Counts the active stance legs
            foot_forces = np.zeros((4, 3))
            stance_count = sum(is_stance_list)

            # NOTE: Calculate the basic static weight distribution 
            # across active stance feet.
            if stance_count > 0:
                weight_per_foot = (self.robot_mass * 9.81) / stance_count
                t_ahead = self.t + i * self.dt
                phase_now = (omega * t_ahead) % (2.0 * m.pi)

                for leg_idx, leg_name in enumerate(LEG_NAMES):
                    leg_phase = (phase_now + self.phase_offsets[leg_name]) % (2.0 * m.pi)
                    if leg_phase < m.pi:
                        # NOTE: calculate the foot forces for going up, applies only 
                        # on the swing phase hence (leg_phase < pi)
                        foot_forces[leg_idx, 2] = weight_per_foot * m.sin(leg_phase)

            # NOTE: Call inverse dynamics to resolve the required joint Torques
            torque = self.inverse_dynamics(q_d, q_dot_d, q_ddot_d, foot_forces, is_stance_list)

            # NOTE: Populate a ROS Trajectory Points to publish to Robot later
            point = JointTrajectoryPoint()
            point.positions = q_d.tolist()
            point.velocities = q_dot_d.tolist()
            point.accelerations = q_ddot_d.tolist()
            point.effort = torque.tolist()
            point.time_from_start = Duration(
                sec=0,
                nanosec=int(i * self.dt * 1e9)
            )
            points.append(point)

        # NOTE: Appends a final anchor point far in the future to satisfy
        # Joint Trajectory Controller
        anchor_point = JointTrajectoryPoint()
        anchor_point.positions = all_pos_history[-1].tolist()
        anchor_point.velocities = np.zeros(12).tolist()
        anchor_point.accelerations = np.zeros(12).tolist()
        anchor_point.effort = torque.tolist()
        anchor_dt = (lookahead_steps + 10) * self.dt
        anchor_point.time_from_start = Duration(
            sec=0,
            nanosec=int(anchor_dt * 1e9)
        )
        points.append(anchor_point)

        # NOTE: publish the points
        msg.points = points
        self.joint_pub.publish(msg)
        self.t += self.dt # increment the time mod (t)

    def jump_process(self, k: KinematicsLogic):
        self.get_logger().warn(f"state : {self.jump_state}")
        y_idle = self.z_off
        y_crouch = self.y_crouch
        y_thrust = self.y_thrust
        y_flight = self.y_flight

        x_idle = self.x_off
        x_thrust = self.x_off + self.x_thrust 
        x_flight = self.x_off + self.x_flight
        x_catch = self.x_off + self.x_catch

        current_y = y_idle
        current_x = x_idle

        avg_fz = np.mean(self.filtered_fz)
        contact_threshold = 2.0

        match self.jump_state:
            case "PREPARE":
                fraction = min(self.t / self.prepare_time, 1.0)
                current_y = y_idle + fraction * (y_crouch - y_idle) 
                current_x = x_idle + fraction * (x_thrust - x_idle)

                t1 = m.degrees(self.current_q[0])
                t2 = m.degrees(self.current_q[1])
                t3 = m.degrees(self.current_q[2])
                fk_matrix = k.fk('FL', t1, t2, t3) 
                current_y_actual = abs(fk_matrix[1, 3])

                if fraction >= 1.0 and abs(current_y_actual - y_crouch) < 0.1:
                    self.jump_state = "THRUST"
                    self.t = 0.0
            case "THRUST":
                fraction = min(self.t / self.thrust_time, 1.0)
                current_y = y_crouch + fraction * (y_thrust - y_crouch)
                current_x = x_thrust 

                if fraction > 0.5 and avg_fz < contact_threshold:
                    self.jump_state = "FLIGHT"
                    self.t = 0.0
                    
            case "FLIGHT":
                fraction = min(self.t / self.flight_time, 1.0)
                current_y = y_thrust + fraction * (y_flight - y_thrust)
                current_x = x_thrust + fraction * (x_flight - x_thrust)

                if fraction >= 1.0:
                    self.jump_state = "DESCENT"
                    self.t = 0.0

            case "DESCENT":
                fraction = min(self.t / self.catch_time, 1.0)
                current_y = y_flight 
                current_x = x_flight + fraction * (x_catch - x_flight)

                if self.t > 0.05 and avg_fz > contact_threshold:
                    self.jump_state = "LANDING"
                    self.t = 0.0

            case "LANDING":
                fraction = min(self.t / self.landing_time, 1.0)
                current_y = y_flight + fraction * (y_idle - y_flight)
                current_x = x_catch + fraction * (x_idle - x_catch)

                if fraction >= 1.0:
                    self.jump_state = ""
                    self.jumping = False
                    if self.jump_timer is not None:
                        self.jump_timer.cancel()
                        self.jump_timer = None
                    return

        msg = JointTrajectory()
        msg.joint_names = []
        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]

        q_desired = []
        for leg in LEG_NAMES:
            ix, iy, iz = k.get_init_pos(leg)
            
            tx = ix + current_x 
            ty = current_y
            tz = iz 
            
            theta1, theta2, theta3 = k.ik(leg, tx, ty, tz)
            q_desired.extend([theta1, theta2, theta3])
            
        graph_msg = Float64MultiArray()
        graph_msg.data = [
            float(self.graph_t),
            float(q_desired[1]),
            float(self.current_q[1])
        ]
        self.graph_pub.publish(graph_msg)

        point = JointTrajectoryPoint()
        point.positions = q_desired
        point.time_from_start = Duration(
            sec=0, 
            nanosec=int(self.dt * 1e9)
        )
        
        msg.points = [point]
        self.joint_pub.publish(msg)
        
        self.t += self.dt
        self.graph_t += self.dt



# NOTE: MAIN DEF 
def main(args=None):
    rclpy.init(args=args)

    node = Tuner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
