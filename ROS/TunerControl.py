import os
import time
import rclpy
import gc

import math as m 
import numpy as np
import pinocchio as pin

from std_msgs.msg import String, Float64MultiArray
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from typing import Optional
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from LOGIC.KinematicsLogic import KinematicsLogic
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

LEG_NAMES = ['FL', 'FR', 'BL', 'BR']

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

class Tuner(Node):
    def __init__(self):
        super().__init__('quins_tuner')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.graph_pub = self.create_publisher(Float64MultiArray, '/tuner/graph', 10)

        self.odometry_subs = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10
        )
        
        self.create_subscription(String, '/tuner/state', self.state_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/params', self.params_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/phase_offsets', self.phase_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/raw', self.raw_tune_callback, 10)

        self.current_yaw = 0.0
        self.current_q = np.zeros(12)
        self.current_q_dot = np.zeros(12)

        self.joint_state_subs = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        self.target_yaw = 0.0
        self.control_rate = 50.0
        self.dt = 1.0 / self.control_rate
        self.t = 0.0
        self.walk_start_time = None
        self.walking = False
        self.walk_timer: Optional[rclpy.timer.Timer] = None
        self.kinematics = KinematicsLogic()

        package_path = get_package_share_directory('quins')
        urdf_path = os.path.join(package_path, 'urdf', 'quadruped.urdf')

        self.pin_model = pin.buildModelsFromUrdf(urdf_path)[0]
        self.pin_data = self.pin_model.createData()
        self.robot_mass = 10.0

        self.current_pos = np.zeros(3)
        self.current_lin_vel = np.zeros(3)
        self.current_ang_vel = np.zeros(3)
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0

        self.last_cycle_wall_time = None
        self.last_join_state_stamp = None

        self.gait_freq = 1.0
        self.x_off = 0.25
        self.z_off = 2.7
        self.step_len = 2.0
        self.step_h = 0.75
        self.sc_yaw = 0.9

        self.phase_offsets = {
            'FL': 0.0,
            'BR': m.pi / 2.0,
            'FR': m.pi,
            'BL': 3.0 * m.pi / 2.0,
        }

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

    def state_callback(self, msg: String):
        self.walking = False
        if hasattr(self, 'walk_timer') and self.walk_timer is not None:
            self.walk_timer.cancel()
            self.walk_timer = None

        match msg.data:
            case "TUNING":
                pass
            case "CROUCH":
                self.animate_transition(0.00, 1.30, -2.70)
            case "IDLE":
                self.animate_transition(0.00, 0.45, 0.90)
            case "WALK":
                self.target_yaw = self.current_yaw
                self.t = 0.0
                self.walk_start_time = self.get_clock().now()
                self.walking = True

                self.walk_timer = self.create_timer(
                    self.dt,
                    lambda: self.walk_process(self.kinematics)
                )

    def params_callback(self, msg: Float64MultiArray):
        self.gait_freq = msg.data[0]
        self.x_off = msg.data[1]
        self.z_off = msg.data[2]
        self.step_len = msg.data[3]
        self.step_h = msg.data[4]
        self.sc_yaw = msg.data[5]

    def phase_callback(self, msg: Float64MultiArray):
        self.phase_offsets = {
            'FL': msg.data[0],
            'BR': msg.data[1],
            'FR': msg.data[2],
            'BL': msg.data[3],
        }

    def raw_tune_callback(self, msg: Float64MultiArray):
        self.send_theta(msg.data[0], msg.data[1], msg.data[2])

    def joint_state_callback(self, msg: JointState):
        expected_order = [
            'tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint',
            'tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint',
            'bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint',
            'br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint',
        ]

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, joint_name in enumerate(expected_order):
            if joint_name in name_to_idx:
                idx = name_to_idx[joint_name]
                self.current_q[i] = msg.position[idx]
                self.current_q_dot[i] = msg.velocity[idx]

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

        for joints in JOINT_NAMES.values():
            msg.joint_names.extend(joints)
            point.positions.extend([coxa, femur, tibia])

        point.time_from_start = Duration(sec=0, nanosec=500000000)
        msg.points.append(point)  # type: ignore
        self.publisher.publish(msg)

    def inverse_dynamics(self, q, q_dot, q_ddot_cmd, foot_forces, is_stance_array):
        m_matrix = pin.crba(self.pin_model, self.pin_data, q)
        bias_forces = pin.rnea(self.pin_model, self.pin_data, q, q_dot, np.zeros_like(q_dot))

        torque = (m_matrix @ q_ddot_cmd) + bias_forces
        foot_frame_names = ['bl_tip_link', 'br_tip_link', 'tl_tip_link', 'tr_tip_link']

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
                
                J_linear = J_full[:3, :]
                torque -= (J_linear.T @ foot_forces[i])

        return torque 

    def trajectory_controller(self, phase, step_len):
        z = 0.0

        if phase < m.pi:
            fraction = phase / m.pi
            x = -(self.step_len / 2.0) + (fraction * step_len)
            y = 0.0
        else:
            fraction = (phase - m.pi) / m.pi
            x = (self.step_len / 2.0) - (fraction * step_len)
            y = self.step_h * m.sin(fraction * m.pi)

        return x, y, z

    def walk_process(self, k: KinematicsLogic):
        if self.walk_start_time is not None:
            self.t = (self.get_clock().now() - self.walk_start_time).nanoseconds * 1e-9
        else:
            self.walk_start_time = self.get_clock().now()
            self.t = 0.0

        omega = 2.0 * m.pi * self.gait_freq

        msg = JointTrajectory()
        msg.joint_names = []

        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]

        lookahead_steps = 10 
        points = []
        all_pos_history = []
        is_stance_history = []

        ramp_duration = 1.0
        ramp_factor = min(self.t / ramp_duration, 1.0)
        base_step_len = self.step_len * ramp_factor

        # Pass 1: Compute target joint positions and stance states
        for i in range(lookahead_steps):
            t_ahead = self.t + (i + 1) * self.dt
            phase_now = (omega * t_ahead) % (2.0 * m.pi)
            q_desired = []
            is_stance_list = []

            yaw_error = self.target_yaw - self.current_yaw

            for leg in LEG_NAMES:
                leg_phase = (phase_now + self.phase_offsets[leg]) % (2.0 * m.pi)
                is_stance = (leg_phase >= m.pi)
                is_stance_list.append(is_stance)
                active_step_len = base_step_len

                if leg in ['FL', 'BL']:
                    active_step_len += (yaw_error * self.sc_yaw) * ramp_factor
                elif leg in ['FR', 'BR']:
                    active_step_len -= (yaw_error * self.sc_yaw) * ramp_factor

                xl, yl, zl = self.trajectory_controller(leg_phase, self.step_len)
                ix, iy, iz = k.get_init_pos(leg)

                tx = ix + xl + self.x_off
                ty = self.z_off - yl
                tz = iz + zl

                theta1, theta2, theta3 = k.ik(leg, tx, ty, tz)
                q_desired.extend([theta1, theta2, theta3])
                
                if i == 0:
                    phi_key = LEG_TO_PHI[leg]
                    self.phi[phi_key]["shoulder"] = theta1
                    self.phi[phi_key]["thigh"] = theta2
                    self.phi[phi_key]["leg"] = theta3

            q_d = np.array(q_desired)
            all_pos_history.append(q_d)
            is_stance_history.append(is_stance_list)

        # Pass 2: Compute analytical velocities (unclamped across active horizon)
        all_vel_history = []
        for i in range(lookahead_steps):
            if i == 0:
                q_dot_d = (all_pos_history[1] - all_pos_history[0]) / self.dt
            elif i == lookahead_steps - 1:
                q_dot_d = (all_pos_history[i] - all_pos_history[i - 1]) / self.dt
            else:
                q_dot_d = (all_pos_history[i + 1] - all_pos_history[i - 1]) / (2.0 * self.dt)
            all_vel_history.append(q_dot_d)

        # Pass 3: Compute analytical accelerations (unclamped across active horizon)
        all_acc_history = []
        for i in range(lookahead_steps):
            if i == 0:
                q_ddot_d = (all_vel_history[1] - all_vel_history[0]) / self.dt
            elif i == lookahead_steps - 1:
                q_ddot_d = (all_vel_history[i] - all_vel_history[i - 1]) / self.dt
            else:
                q_ddot_d = (all_vel_history[i + 1] - all_vel_history[i - 1]) / (2.0 * self.dt)
            all_acc_history.append(q_ddot_d)

        # Pass 4: Compute inverse dynamics torques and construct active trajectory points
        torque = np.zeros(14) 
        for i in range(lookahead_steps):
            q_d = all_pos_history[i]
            q_dot_d = all_vel_history[i]
            q_ddot_d = all_acc_history[i]
            is_stance_list = is_stance_history[i]

            if i == 0:
                q_eval = self.current_q
                q_dot_eval = self.current_q_dot

                graph_msg = Float64MultiArray()
                graph_msg.data = [
                    float(self.t),
                    float(q_d[1]),
                    float(self.current_q[1])
                ]
                self.graph_pub.publish(graph_msg)

                now = self.get_clock().now()
                staleness = 0.0
                if self.last_cycle_wall_time is not None:
                    actual_dt = (now - self.last_cycle_wall_time).nanoseconds * 1e-9
                    if abs(actual_dt - self.dt) > 1.5 * self.dt:
                        self.get_logger().warn(
                            f"fuckign jitter: expected {self.dt*1000:.1f} ms, "
                            f"got {actual_dt*1000:.1f} ms"
                        )
                self.last_cycle_wall_time = now

                if self.last_join_state_stamp is not None:
                    staleness = (now - self.last_join_state_stamp).nanoseconds * 1e-9
                    if staleness > 2.5 * self.dt:
                        self.get_logger().warn(
                            f"current_q is {staleness*1000:.1f} ms stale at consumption"
                        )
                # if staleness < 0.10:
                #     q_eval = self.current_q
                #     q_dot_eval = self.current_q_dot
                #
                #     kp = 100.0
                #     kd = 10.0
                #     q_ddot_cmd = q_ddot_d + kp * (q_d - q_eval) + kd * (q_dot_d - q_dot_eval)
                # else:
                #     q_eval = q_d
                #     q_dot_eval = q_dot_d
                #     q_ddot_cmd = q_ddot_d

            foot_forces = np.zeros((4, 3))
            stance_count = sum(is_stance_list)
            if stance_count > 0:
                weight_per_foot = (self.robot_mass * 9.81) / stance_count
                t_ahead = self.t + (i + 1) * self.dt
                phase_now = (omega * t_ahead) % (2.0 * m.pi)

                for leg_idx, leg_name in enumerate(LEG_NAMES):
                    leg_phase = (phase_now + self.phase_offsets[leg_name]) % (2.0 * m.pi)
                    if leg_phase < m.pi:
                        foot_forces[leg_idx, 2] = weight_per_foot * m.sin(leg_phase)

            torque = self.inverse_dynamics(q_d, q_dot_d, q_ddot_d, foot_forces, is_stance_list)

            point = JointTrajectoryPoint()
            point.positions = q_d.tolist()
            point.velocities = q_dot_d.tolist()
            point.accelerations = q_ddot_d.tolist()
            point.effort = torque.tolist()
            point.time_from_start = Duration(
                sec=0,
                nanosec=int((i + 1) * self.dt * 1e9)
            )
            points.append(point)

        # Pass 5: Append safety anchor point to satisfy ROS 2 JTC zero-velocity end requirement
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

        msg.points = points
        self.publisher.publish(msg)
        self.t += self.dt

    def animate_transition(self, target_s, target_t, target_k, duration=1.0):
        if hasattr(self, 'transition_timer') and self.transition_timer is not None:
            self.transition_timer.cancel()
            self.transition_timer = None

        start_time = time.time()
        initial_s = self.phi["tr_leg"]["shoulder"]
        initial_t = self.phi["tr_leg"]["thigh"]
        initial_k = self.phi["tr_leg"]["leg"]

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

        self.transition_timer = self.create_timer(0.02, step)


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
