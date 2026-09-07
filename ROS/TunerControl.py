import os
import rclpy
import math as m 
import numpy as np
import pinocchio as pin

from rclpy.node import Node
from nav_msgs.msg import Odometry 
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Float64MultiArray 
from ament_index_python.packages import get_package_share_directory

from LOGIC.GaitLogic import GaitLogic, LEG_NAMES, JOINT_NAMES
from LOGIC.FOSMCLogic import FOSMC
from LOGIC.MpcLogic import CentroidalMPC

class Tuner(Node):
    def __init__(self):
        super().__init__('quins_tuner')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', Parameter.Type.BOOL, True)]) 

        # NOTE: Publishers 
        self.joint_pub = self.create_publisher(JointState, '/isaac_joint_commands', 10)
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
        urdf_path = os.path.join(package_path, 'urdf', 'quadruped_isaac.urdf')
        # workspace_path = os.path.expanduser('~/ros2_ws/src')
        self.pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.pin_data = self.pin_model.createData()

        # NOTE: Floating Base State Storage for MPC
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0]) # x, y, z, w
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        # NOTE: MPC n FOSMC Param
        self.mpc = CentroidalMPC(urdf_path, n=10, dt=self.dt)
        self.fosmc = FOSMC(
            dof=12,
            dt=self.dt,
            lam=0.5,
            alpha=1.5,
            Ke1=24.0,
            Ke2=0.5,
            Ks=2.5,
            Kr=1.0,
            gamma_c=0.01,
            gamma_a=0.01
        )

        self.mpc_counter = 0
        self.optimal_foot_forces = np.zeros(12)

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

        self.gait_logic.loop_step()

    def odometry_callback(self, msg: Odometry):
        self.base_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.base_lin_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.base_ang_vel = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        
        q = msg.pose.pose.orientation
        self.base_quat = np.array([q.x, q.y, q.z, q.w])

        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.gait_logic.current_roll = m.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (qw * qy - qz * qx)
        self.gait_logic.current_pitch = m.asin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.gait_logic.current_yaw = m.atan2(siny_cosp, cosy_cosp)
        self.get_logger().info(f"Current Physical Z Height: {self.base_pos[2]:.3f} m\n")

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
        self.publish_positions(positions)

    def handle_transition(self, current_angles):
        self.publish_positions(current_angles)

    def handle_jump_points(self, q_desired):
        self.publish_positions(q_desired)

    def handle_walk_points(self, points_data):
        if not points_data:
            return

        pt = points_data[0]
        q_d = np.array(pt["positions"])
        q_dot_d = np.array(pt["velocities"])
        is_stance = pt["is_stance"]

        q_act = self.gait_logic.current_q
        qd_act = self.gait_logic.current_q_dot
        rpy_act = np.array([self.gait_logic.current_roll, self.gait_logic.current_pitch, self.gait_logic.current_yaw])

        expected_order = [
            'tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint',
            'tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint',
            'bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint',
            'br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint',
        ]

        # 1. Assemble Full FreeFlyer State Vector (Size 19) for Pinocchio
        q_full = np.zeros(19)
        q_full[0:3] = self.base_pos
        q_full[3:7] = self.base_quat
        
        # Explicitly map your ROS array to Pinocchio's internal index order
        for i, joint_name in enumerate(expected_order):
            if self.pin_model.existJointName(joint_name):
                jid = self.pin_model.getJointId(joint_name)
                idx_q = self.pin_model.joints[jid].idx_q
                q_full[idx_q] = q_act[i]

        pin.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # 2. Extract Cartesian Foot Positions for MPC
        foot_positions = []
        foot_frame_names = ['tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link']
        for name in foot_frame_names:
            fid = self.pin_model.getFrameId(name)
            foot_positions.append(self.pin_data.oMf[fid].translation)

        # 3. Solve Whole-Body MPC (Decimated to free up the control thread)
        self.mpc_counter += 1
        
        if self.mpc_counter % 5 == 0:
            planned_stance_schedule = [is_stance for _ in range(self.mpc.n)]
            opti, U_f, p_ref, v_ref = self.mpc.setup_problem(
                self.base_pos, self.base_lin_vel, rpy_act, self.base_ang_vel, 
                foot_positions, planned_stance_schedule
            )

            target_z = getattr(self.gait_logic, 'z_off_used', 2.55)
            opti.set_value(p_ref, np.array([self.base_pos[0], self.base_pos[1], target_z]))
            
            # Allow the MPC to push forward if the gait is walking
            target_vx = 1.0 if self.gait_logic.current_state == "WALK" else 0.0
            opti.set_value(v_ref, np.array([target_vx, 0.0, 0.0]))

            try:
                sol = opti.solve()
                self.optimal_foot_forces = np.array(sol.value(U_f[:, 0])).flatten()
            except:
                pass 
                
        # Use the persistent optimal_foot_forces for the rest of the loop
        optimal_foot_forces = self.optimal_foot_forces

        # 4. Fractional-Order Sliding Mode Control (Feedback)
        fosmc_out = self.fosmc.compute(q_act, qd_act, q_d, q_dot_d)
        tau_pd = fosmc_out[0] if isinstance(fosmc_out, tuple) else fosmc_out
        tau_pd = np.array(tau_pd).flatten()
        tau_pd = np.nan_to_num(tau_pd, nan=0.0, posinf=1500.0, neginf=-1500.0)

        # 5. Whole-Body Inverse Dynamics (Feedforward Mapping)
        tau_ff_pin = np.zeros(18) 
        pin.computeJointJacobians(self.pin_model, self.pin_data, q_full)

        for i, name in enumerate(foot_frame_names):
            if is_stance[i]:
                fid = self.pin_model.getFrameId(name)
                J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_full, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_linear = J[:3, :] 
                F_i = optimal_foot_forces[3*i : 3*i+3]
                tau_ff_pin -= J_linear.T @ F_i  # This MUST be subtraction

        # Explicitly map Pinocchio's 18D torques back to your 12D ROS array order
        tau_ff = np.zeros(12)
        for i, joint_name in enumerate(expected_order):
            if self.pin_model.existJointName(joint_name):
                jid = self.pin_model.getJointId(joint_name)
                idx_v = self.pin_model.joints[jid].idx_v
                tau_ff[i] = tau_ff_pin[idx_v]

        # Unified Control Output
        final_torque = np.clip(tau_ff + tau_pd, -1500.0, 1500.0)

        # DEBUG LOGGER: Print the Front-Left (FL) Leg calculations
        # FL mapped indices: Shoulder=0, Thigh=1, Calf=2
        self.get_logger().info(
            f"\n--- FL LEG DEBUG ---\n"
            f"MPC Target Z Force: {optimal_foot_forces[2]:.2f} N\n"
            f"Feedforward Torque (Shoulder, Thigh, Calf): {tau_ff[0]:.2f}, {tau_ff[1]:.2f}, {tau_ff[2]:.2f}\n"
            f"FOSMC Torque (Shoulder, Thigh, Calf): {tau_pd[0]:.2f}, {tau_pd[1]:.2f}, {tau_pd[2]:.2f}\n"
            f"Final Output: {final_torque[0]:.2f}, {final_torque[1]:.2f}, {final_torque[2]:.2f}\n"
        )

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = []
        for leg in LEG_NAMES:
            msg.name += JOINT_NAMES[leg]

        msg.position = q_d.tolist()
        msg.velocity = q_dot_d.tolist()
        msg.effort = final_torque.tolist()
        self.joint_pub.publish(msg)

    def publish_positions(self, positions_12):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = []

        for leg in LEG_NAMES:
            msg.name += JOINT_NAMES[leg]

        msg.position = positions_12
        msg.velocity = np.zeros(12).tolist()
        msg.effort = np.zeros(12).tolist()

        self.joint_pub.publish(msg)

    # NOTE: -------- PINOCCHIO (Framework Specific) --------
    def inverse_dynamics(self, q, q_dot, q_ddot_cmd, foot_forces, is_stance_array):
        m_matrix = pin.crba(self.pin_model, self.pin_data, q)
        bias_forces = pin.rnea(self.pin_model, self.pin_data, q, q_dot, np.zeros_like(q_dot))
        torque = (m_matrix @ q_ddot_cmd) + bias_forces

        foot_frame_names = ['fl_tip_link', 'fr_tip_link', 'bl_tip_link', 'br_tip_link']

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
