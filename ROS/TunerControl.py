import os
import rclpy
import math as m 
import numpy as np
import pinocchio as pin

from rclpy.node import Node
from nav_msgs.msg import Odometry 
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Float64MultiArray 
from ament_index_python.packages import get_package_share_directory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES

class Tuner(Node):
    def __init__(self):
        super().__init__('quins_tuner')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', Parameter.Type.BOOL, True)]) 

        # NOTE: Publishers 
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.graph_pub = self.create_publisher(Float64MultiArray, '/tuner/graph', 10)

        # NOTE: Setup GaitLogic Callbacks
        callbacks = {
            "walk_points": self.handle_walk_points,
            "jump_points": self.handle_jump_points,
            "transition_cb": self.handle_transition,
            "raw_tune_cb": self.handle_raw_tune,
            "graph": self.handle_graph
        }
        
        # Instantiate the pure math logic
        self.gait_logic = GaitLogic(callbacks=callbacks)

        # NOTE: Subscribers
        self.create_subscription(Odometry, '/model/quadruped/odometry', self.odometry_callback, 10)
        self.create_subscription(String, '/tuner/state', self.state_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/raw', self.raw_tune_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/params', self.params_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/jparams', self.jump_params_callback, 10)
        self.create_subscription(Float64MultiArray, '/tuner/phase_offsets', self.phase_callback, 10)

        # NOTE: Timer (ROS2 drives the math loop)
        self.control_rate = 50.0 
        self.dt = 1.0 / self.control_rate 
        self.create_timer(self.dt, self.gait_logic.loop_step)

        # NOTE: Pinocchio Model (Framework specific physics)
        package_path = get_package_share_directory('quins')
        urdf_path = os.path.join(package_path, 'urdf', 'quadruped.urdf')
        self.pin_model = pin.buildModelsFromUrdf(urdf_path)[0]
        self.pin_data = self.pin_model.createData()

    # NOTE: -------- Telemetry Sync -> GaitLogic --------
    def joint_state_callback(self, msg: JointState):
        expected_order = [
            'tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint',
            'tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint',
            'bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint',
            'br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint',
        ]

        q_sorted = np.zeros(12)
        q_dot_sorted = np.zeros(12)

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, joint_name in enumerate(expected_order):
            if joint_name in name_to_idx:
                idx = name_to_idx[joint_name]
                q_sorted[i] = msg.position[idx]
                q_dot_sorted[i] = msg.velocity[idx]

        self.gait_logic.current_q = q_sorted
        self.gait_logic.current_q_dot = q_dot_sorted

    def odometry_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.gait_logic.current_roll = m.atan2(sinr_cosp, cosr_cosp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.gait_logic.current_yaw = m.atan2(siny_cosp, cosy_cosp)

    # NOTE: -------- Incoming GUI Parameters -> GaitLogic --------
    def state_callback(self, msg: String):
        self.gait_logic.update_state(msg.data)

    def params_callback(self, msg: Float64MultiArray):
        self.gait_logic.update_wt_params(list(msg.data))

    def jump_params_callback(self, msg: Float64MultiArray):
        self.gait_logic.update_jt_params(list(msg.data))

    def phase_callback(self, msg: Float64MultiArray):
        self.gait_logic.update_phase_offsets(list(msg.data))

    def raw_tune_callback(self, msg: Float64MultiArray):
        self.gait_logic.raw_tune(list(msg.data))

    # NOTE: -------- Outbound Math -> ROS2 Messages --------
    def handle_graph(self, graph_data):
        msg = Float64MultiArray()
        msg.data = graph_data
        self.graph_pub.publish(msg)

    def handle_raw_tune(self, raw_angles):
        coxa, femur, tibia = raw_angles
        positions = []
        for _ in range(4): # Append the same 3 angles for all 4 legs
            positions.extend([coxa, femur, tibia])
        self.publish_positions(positions, 0.5)

    def handle_transition(self, current_angles):
        self.publish_positions(current_angles, 0.02)

    def handle_jump_points(self, q_desired):
        self.publish_positions(q_desired, self.dt)

    def handle_walk_points(self, points_data):
        msg = JointTrajectory()
        msg.joint_names = []
        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]

        points = []
        for pt in points_data:
            q_d = np.array(pt["positions"])
            q_dot_d = np.array(pt["velocities"])
            q_ddot_d = np.array(pt["accelerations"])
            foot_forces = np.array(pt["foot_forces"])
            is_stance = pt["is_stance"]

            # Wrap logic purely through Pinocchio ID for this specific env
            torque = self.inverse_dynamics(q_d, q_dot_d, q_ddot_d, foot_forces, is_stance)

            point = JointTrajectoryPoint()
            point.positions = q_d.tolist()
            point.velocities = q_dot_d.tolist()
            point.accelerations = q_ddot_d.tolist()
            point.effort = torque.tolist()
            point.time_from_start = Duration(sec=0, nanosec=int(pt["time_offset"] * 1e9))
            points.append(point)

        # Append anchor point (far into the future to satisfy ROS2 controller)
        if points:
            anchor = JointTrajectoryPoint()
            anchor.positions = points[-1].positions
            anchor.velocities = np.zeros(12).tolist()
            anchor.accelerations = np.zeros(12).tolist()
            anchor.effort = points[-1].effort
            
            # Explicitly reference the last item in the array instead of the loop variable
            last_time_offset = points_data[-1]["time_offset"]
            anchor_dt = last_time_offset + (10 * self.dt)
            
            anchor.time_from_start = Duration(sec=0, nanosec=int(anchor_dt * 1e9))
            points.append(anchor)

        msg.points = points
        self.joint_pub.publish(msg)

    def publish_positions(self, positions_12, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = []
        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]
            
        point = JointTrajectoryPoint()
        point.positions = positions_12
        point.time_from_start = Duration(
            sec=int(duration_sec), 
            nanosec=int((duration_sec % 1) * 1e9)
        )
        
        msg.points.append(point)
        self.joint_pub.publish(msg)

    # NOTE: -------- PINOCCHIO (Framework Specific) --------
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
