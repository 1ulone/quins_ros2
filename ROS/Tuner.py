import os
import time
import rclpy
import threading

import math as m 
import numpy as np
import tkinter as tk
from tkinter import ttk
import pinocchio as pin

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from typing import Optional
from collections import deque
from tkinter.font import Font
from nav_msgs.msg import Odometry 
from matplotlib.figure import Figure
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from LOGIC.KinematicsLogic import KinematicsLogic
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.current_yaw = 0.0
        self.odometry_subs = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10
        )

        self.current_q = np.zeros(12)
        self.current_q_dot = np.zeros(12)

        self.joint_state_subs = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

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

        self.phase_offsets = {
            'FL': 0.0,
            'BR': m.pi / 2.0,
            'FR': m.pi,
            'BL': 3.0 * m.pi / 2.0,
        }

        self.gait_freq = 1.0
        self.x_off = 0.25
        self.z_off = 2.7
        self.step_len = 2.0
        self.step_h = 0.75
        self.sc_yaw = 0.9

        self.target_yaw = 0.0
        self.control_rate = 50.0
        self.dt = 1.0 / self.control_rate
        self.t = 0.0
        self.walk_start_time = None
        self.walking = False
        self.walk_timer: Optional[rclpy.timer.Timer] = None

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

        self.time_history = deque(maxlen=200)
        self.desired_history = deque(maxlen=200)
        self.measured_history = deque(maxlen=200)

        self.last_cycle_wall_time = None
        self.last_join_state_stamp = None

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
        # if self.walk_start_time is not None:
        #     self.t = (self.get_clock().now() - self.walk_start_time).nanoseconds * 1e-9
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

            if i == 0:
                is_stance_list = is_stance_history[0]
                q_eval = self.current_q
                q_dot_eval = self.current_q_dot
                self.time_history.append(self.t)
                self.desired_history.append(q_d[1])
                self.measured_history.append(self.current_q[1])

                now = self.get_clock().now()
                if self.last_cycle_wall_time is not None:
                    actual_dt = (now - self.last_cycle_wall_time).nanoseconds * 1e-9
                    if abs(actual_dt - self.dt) > 0.2 * self.dt:
                        self.get_logger().warn(
                            f"fuckign jitter: expected {self.dt*1000:.1f} ms, "
                            f"got {actual_dt*1000:.1f} ms"
                        )
                self.last_cycle_wall_time = now

                if self.last_join_state_stamp is not None:
                    staleness = (now - self.last_join_state_stamp).nanoseconds * 1e-9
                    if staleness > 0.5 * self.dt:
                        self.get_logger().warn(
                            f"current_q is {staleness*1000:.1f} ms stale at consumption"
                        )

                kp = 100.0
                kd = 10.0
                q_ddot_cmd = q_ddot_d + kp * (q_d - q_eval) + kd * (q_dot_d - q_dot_eval)
                foot_forces = np.zeros((4, 3))
                stance_count = sum(is_stance_list)
                if stance_count > 0:
                    weight_per_foot = (self.robot_mass * 9.81) / stance_count
                    for leg_idx, in_stance in enumerate(is_stance_list):
                        if in_stance:
                            foot_forces[leg_idx, 2] = weight_per_foot

                torque = self.inverse_dynamics(q_eval, q_dot_eval, q_ddot_cmd, foot_forces, is_stance_list)

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


# NOTE: MAIN DEF 
def main(args=None):
    rclpy.init(args=args)
    import gc
    gc.disable()
    node = Tuner()
    kinematics = KinematicsLogic()

    def lerp(a, b, t):
        return a + t * (b - a)

    def logic_process(s, t, k):
        node.send_theta(s, t, k)

    def tuning_process(val):
        s = float(shoulder_slider.get())
        t = float(thigh_slider.get())
        k = float(leg_slider.get())

        logic_process(s, t, k)
        animate_transition(s, t, k, 0.75)

    def animate_transition(target_s, target_t, target_k, duration=1.0):
        start_time = time.time()
        initial_s = node.phi["tr_leg"]["shoulder"]
        initial_t = node.phi["tr_leg"]["thigh"]
        initial_k = node.phi["tr_leg"]["leg"]

        def step():
            elapsed = time.time() - start_time
            fraction = min(elapsed / duration, 1.0)

            current_s = lerp(initial_s, target_s, fraction)
            current_t = lerp(initial_t, target_t, fraction)
            current_k = lerp(initial_k, target_k, fraction)

            for leg in node.phi.values():
                leg["shoulder"] = current_s
                leg["thigh"] = current_t 
                leg["leg"] = current_k 

            if fraction < 1.0:
                root.after(20, step)

        step()

    root = tk.Tk()
    fig = Figure(figsize=(7.5, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Joint Position Tracking")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.NONE, expand=False, pady=5)

    def refresh_plot():
        if len(node.time_history) > 1 and node.walking:
            ax.clear()
            
            ax.plot(list(node.time_history), list(node.desired_history), 'k--', label="Desired angle")
            ax.plot(list(node.time_history), list(node.measured_history), 'y-', label="Measured angle")
            
            ax.set_title("Joint Position Tracking (tl_shoulder_joint)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (rad)")
            ax.legend(loc="upper right")
            ax.grid(True)
            
            canvas.draw()
            
        root.after(500, refresh_plot)

    root.after(500, refresh_plot)

    root.title("Quadruped Tuner")
    root.geometry("500x450") 

    font_style = Font(family="Arial", size=10, weight="bold") 

    grid_container = tk.Frame(root)
    grid_container.pack(pady=10, padx=10)

    state = tk.StringVar(value="TUNING")
    state_group = ttk.LabelFrame(grid_container, text=f"Change State {state.get()}", padding=15)
    state_group.grid(row=0, column=0, sticky='nsew')

    def on_tuning(is_on):
        if is_on:
            shoulder_slider.pack()
            shoulder_slider_label.pack()
            thigh_slider.pack()
            thigh_slider_label.pack()
            leg_slider.pack()
            leg_slider_label.pack()
        else:
            shoulder_slider.pack_forget()
            shoulder_slider_label.pack_forget()
            thigh_slider.pack_forget()
            thigh_slider_label.pack_forget()
            leg_slider.pack_forget()
            leg_slider_label.pack_forget()

    def update_state():
        state_group.config(text=f"Change State {state.get()}")
        node.walking = False

        if hasattr(node, 'walk_timer') and node.walk_timer is not None:
            node.walk_timer.cancel()
            node.walk_timer = None

        match state.get():
            case "TUNING":
                on_tuning(True)
                shoulder_slider.set(node.phi["tr_leg"]["shoulder"])
                thigh_slider.set(node.phi["tr_leg"]["thigh"])
                leg_slider.set(node.phi["tr_leg"]["leg"])
            case "CROUCH":
                on_tuning(False)
                logic_process(0, 1.30, -2.70)
                animate_transition(0, 0.75, -1.50, duration=1.0)
            case "IDLE":
                on_tuning(False)
                logic_process(0, 0.45, -0.90)
                animate_transition(0, 0, 0, duration=1.0)
            case "WALK":
                on_tuning(False)
                node.target_yaw = node.current_yaw
                node.t = 0.0
                node.walk_start_time = node.get_clock().now()
                node.walking = True

                node.walk_timer = node.create_timer(
                    node.dt, 
                    lambda: node.walk_process(kinematics)
                )

    tuningBtn = tk.Radiobutton(
        state_group,
        text="TUNING",
        value="TUNING",
        variable=state,
        font=font_style,
        command=update_state,
    )
    tuningBtn.grid(column=0, row=0)

    crouchBtn = tk.Radiobutton(
        state_group,
        text="CROUCH",
        value="CROUCH",
        variable=state,
        font=font_style,
        command=update_state,
    )
    crouchBtn.grid(column=1, row=0)

    idleBtn = tk.Radiobutton(
        state_group,
        text="IDLE",
        value="IDLE",
        variable=state,
        font=font_style,
        command=update_state,
    )
    idleBtn.grid(column=0, row=1)

    walkBtn = tk.Radiobutton(
        state_group,
        text="WALK",
        value="WALK",
        variable=state,
        font=font_style,
        command=update_state,
    )
    walkBtn.grid(column=1, row=1)

    phase_s = tk.StringVar(value="CROSS")
    phase_group = ttk.LabelFrame(grid_container, text=f"Change Phase Type {phase_s.get()}", padding=15)
    phase_group.grid(row=0, column=1, sticky='nsew')

    def update_phase():
        phase_group.config(text=f"Change State {phase_s.get()}")

        match phase_s.get():
            case "CROSS":
                node.phase_offsets = {
                    'FL': 0.0,
                    'BR': 0.0,
                    'FR': m.pi,
                    'BL': m.pi,
                }
            case "4BEAT":
                node.phase_offsets = {
                    'FL': 0.0,
                    'BR': m.pi / 2.0,
                    'FR': m.pi,
                    'BL': 3.0 * m.pi / 2.0,
                }

    cross = tk.Radiobutton(
        phase_group,
        text="Cross Pair",
        value="CROSS",
        variable=phase_s,
        font=font_style,
        command=update_phase,
    )
    cross.pack()

    creep = tk.Radiobutton(
        phase_group,
        text="4-beat cycle",
        value="4BEAT",
        variable=phase_s,
        font=font_style,
        command=update_phase,
    )
    creep.pack()

    wt_group = ttk.LabelFrame(grid_container, text="WALK TUNING PARAMS", padding=15)
    wt_group.grid(row=1, column=0, columnspan=2)

    gf_frame = ttk.Frame(wt_group)
    gf_frame.grid(row=0, column=0)
    tk.Label(gf_frame, text="Gait Frequency").pack()
    gf = tk.StringVar(value=str(node.gait_freq))
    tk.Entry(gf_frame, textvariable=gf).pack()

    xo_frame = ttk.Frame(wt_group)
    xo_frame.grid(row=0, column=1)
    tk.Label(xo_frame, text="X Offset (Forward/Back Shift)").pack()
    xo = tk.StringVar(value=str(node.x_off))
    tk.Entry(xo_frame, textvariable=xo).pack()

    zo_frame = ttk.Frame(wt_group)
    zo_frame.grid(row=0, column=2)
    tk.Label(zo_frame, text="Z Offset (Ride Height)").pack()
    zo = tk.StringVar(value=str(node.z_off))
    tk.Entry(zo_frame, textvariable=zo).pack()

    sl_frame = ttk.Frame(wt_group)
    sl_frame.grid(row=1, column=0)
    tk.Label(sl_frame, text="Step Length (Stride)").pack()
    sl = tk.StringVar(value=str(node.step_len))
    tk.Entry(sl_frame, textvariable=sl).pack()

    sh_frame = ttk.Frame(wt_group)
    sh_frame.grid(row=1, column=1)
    tk.Label(sh_frame, text="Step Height (Clearance)").pack()
    sh = tk.StringVar(value=str(node.step_h))
    tk.Entry(sh_frame, textvariable=sh).pack()

    sc_frame = ttk.Frame(wt_group)
    sc_frame.grid(row=1, column=2)
    tk.Label(sc_frame, text="Self Correcting Yaw Feedback").pack()
    sc = tk.StringVar(value=str(node.sc_yaw))
    tk.Entry(sc_frame, textvariable=sc).pack()

    def apply_entry(event=None):
        try:
            node.gait_freq = float(gf.get())
            node.x_off = float(xo.get())
            node.z_off = float(zo.get()) 
            node.step_len = float(sl.get())
            node.step_h = float(sh.get())
            node.sc_yaw = float(sc.get())
        except ValueError:
            pass

    apply_btn = tk.Button(wt_group, text="Apply", command=apply_entry)
    apply_btn.grid(row=2, column=0, columnspan=3, sticky='nsew')

    shoulder_slider_label = tk.Label(root, text="Shoulder Angle", font=font_style)
    shoulder_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    thigh_slider_label = tk.Label(root, text="Thigh Angle", font=font_style)
    thigh_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    leg_slider_label = tk.Label(root, text="Knee Angle", font=font_style)
    leg_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    # spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
