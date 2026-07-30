import os
import rclpy
import threading
import time
import math as m 
import numpy as np
import tkinter as tk
import pinocchio as pin
from typing import Optional
from tkinter.font import Font
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from LOGIC.KinematicsLogic import KinematicsLogic
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState

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

        # self.mqtt = MqttLogic()
        # self.timer = self.create_timer(2.0, self.send_mqtt)

        self.gait_freq = 2.5
        self.x_off = 0.25
        self.z_off = 2.7
        self.step_len = 2.0
        self.step_h = 0.75
        self.sc_yaw = 0.9

        self.target_yaw = 0.0
        self.control_rate = 50.0
        self.dt = 1.0 / self.control_rate
        self.t = 0.0
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
        msg.points.append(point) # type: ignore
        self.publisher.publish(msg)

    # def send_mqtt(self):
    #     self.mqtt.client.publish(self.mqtt.main_topic, json.dumps(self.phi))

    def inverse_dynamics(self, q, q_dot, q_ddot_cmd, foot_forces, is_stance_array):
        # NOTE: skipped eq 12 through 14, just a tad bit different in some equation
        # but nonetheless same result of the constraint matrix.
        m = pin.crba(self.pin_model, self.pin_data, q)
        bias_forces = pin.rnea(self.pin_model, self.pin_data, q, q_dot, np.zeros_like(q_dot))

        # NOTE: uses the main equation of dynamics.
        # Torque = M(q)qdot + C(q,qdot) + G(q) - JcFc
        torque = (m @ q_ddot_cmd) + bias_forces
        foot_frame_names = ['bl_tip_link', 'br_tip_link', 'tl_tip_link', 'tr_tip_link']

        for i,frame_name in enumerate(foot_frame_names):
            if is_stance_array[i]:
                frame_id = self.pin_model.getFrameId(frame_name)

                # NOTE: result in Spatial Contact Jacobian Matrix 6x12 
                #  6x12 matrix but really a nothing burger since only 6 values is 
                #  needed/used, the top 3 (linear velocity) and the bottom 
                #  (angular velocity).
                J_full = pin.computeFrameJacobian(
                    self.pin_model, 
                    self.pin_data, 
                    q, 
                    frame_id, 
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                
                # NOTE: Extract linear XYZ translation Jacobian (first 3 rows)
                # foot point on making contact only uses 'linear' pushing forces
                J_linear = J_full[:3, :]
                
                # NOTE: The Error of J^Tc.Fc
                torque -= (J_linear.T @ foot_forces[i])

        return torque 

    def trajectory_controller(self, phase, step_len):
        z = 0.0

        if phase < m.pi:
            # STANCE
            fraction = phase / m.pi
            x = -(self.step_len / 2.0) + (fraction * step_len)
            y = 0.0
        else:
            # SWING
            fraction = (phase - m.pi) / m.pi
            x = (self.step_len / 2.0) - (fraction * step_len)
            y = self.step_h * m.sin(fraction * m.pi)

        return x, y, z

    def walk_process(self, k: KinematicsLogic):
        omega = 2.0 * m.pi * self.gait_freq

        msg = JointTrajectory()
        msg.joint_names = []

        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]

        # NOTE: make the Controller to look ahead of what steps to follow
        # instead of a single fixed target
        lookahead_steps = 5
        points = []

        ramp_duration = 1.0
        ramp_factor = min(self.t / ramp_duration, 1.0)
        base_step_len = self.step_len * ramp_factor

        for i in range(lookahead_steps):
            t_ahead = self.t + i * self.dt
            phase_now = (omega * t_ahead) % (2.0 * m.pi)
            all_pos = []
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
                    T = k.fk(leg, m.degrees(theta1),  m.degrees(theta2), m.degrees(theta3))
                    self.get_logger().info(
                        f"[{leg}] {'SWING ' if leg_phase < m.pi else 'STANCE'} | "
                        f"θ=({theta1:.1f}, {theta2:.1f}, {theta3:.1f}) | "
                        f"foot=({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})"
                    )

                    # for sending to MQTT 
                    phi_key = LEG_TO_PHI[leg]
                    self.phi[phi_key]["shoulder"] = theta1
                    self.phi[phi_key]["thigh"] = theta2
                    self.phi[phi_key]["leg"] = theta3

                all_pos += [theta1, theta2, theta3]

            q_d = np.array(q_desired)
            q_curr = self.current_q 
            q_dot_curr = self.current_q_dot

            kp = 100.0
            kd = 10.0

            q_ddot_cmd = kp * (q_d - q_curr) + kd * (0.0 - q_dot_curr)

            foot_forces = np.zeros((4, 3))
            stance_count = sum(is_stance_list)
            if stance_count > 0:
                weight_per_foot = (self.robot_mass * 9.81) / stance_count
                for leg_idx, in_stance in enumerate(is_stance_list):
                    if in_stance:
                        foot_forces[leg_idx, 2] = weight_per_foot

            torque = self.inverse_dynamics(q_curr, q_dot_curr, q_ddot_cmd, foot_forces, is_stance_list)

            point = JointTrajectoryPoint()
            point.positions = all_pos
            
            point.effort = torque.tolist()

            point.time_from_start = Duration(
                sec=0,
                nanosec=int((i + 1) * self.dt * 1e9)
            )
            points.append(point)

        msg.points = points
        self.publisher.publish(msg)
        self.t += self.dt

def main(args=None):
    rclpy.init(args=args)
    node = Tuner()
    kinematics = KinematicsLogic()

    def lerp(a, b, t):
        return a + t * (b - a)


    def logic_process(s, t, k):
        node.send_theta(s, t, k)

        t04 = kinematics.fk('FL', m.degrees(s), m.degrees(t), m.degrees(k))
        x = t04[0, 3]
        y = t04[1, 3]
        z = t04[2, 3]

        rad_label.config(text=f"Radian Angle:\n[ COXA:{s:.2f}, TIBIA:{t:.2f}, FEMUR:{k:.2f} ]")
        deg_label.config(text=f"Degree's Angle:\n[ COXA:{m.degrees(s):.2f}, TIBIA:{m.degrees(t):.2f}, FEMUR:{m.degrees(k):.2f} ]")
        xyz_label.config(text=f"XYZ Position:\n[ X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f} ]")


    def tuning_process(val):
        s = float(shoulder_slider.get())
        t = float(thigh_slider.get())
        k = float(leg_slider.get())

        logic_process(s, t, k)
        animate_transition(s, t, k, 0.75)


    def animate_transition(target_s, target_t, target_k, duration=1.0):
        start_time = time.time()
        # Snapshot the starting position
        initial_s = node.phi["tr_leg"]["shoulder"]
        initial_t = node.phi["tr_leg"]["thigh"]
        initial_k = node.phi["tr_leg"]["leg"]

        def step():
            elapsed = time.time() - start_time
            fraction = min(elapsed / duration, 1.0) # Clamps at 1.0 (100%)

            current_s = lerp(initial_s, target_s, fraction)
            current_t = lerp(initial_t, target_t, fraction)
            current_k = lerp(initial_k, target_k, fraction)

            for leg in node.phi.values():
                leg["shoulder"] = current_s
                leg["thigh"] = current_t 
                leg["leg"] = current_k 

            if fraction < 1.0:
                root.after(20, step) # Execute step() again in 20ms

        step() # Start the loop

    root = tk.Tk()
    root.title("Quadruped Tuner")
    root.geometry("500x450") 

    font_style = Font(family="Arial", size=10, weight="bold") 

    # NOTE: 
    # ----- POSITION AND ANGLE ------

    rad_label = tk.Label(
        root,
        text="Radian Angle:\n[ COXA:0.00, TIBIA:0.00, FEMUR:0.00 ]",
        font=("Courier", 12, "bold"),
        fg="red"
    )
    deg_label = tk.Label(
        root,
        text="Degree's Angle:\n[ COXA:0.00, TIBIA:0.00, FEMUR:0.00 ]",
        font=("Courier", 12, "bold"),
        fg="red"
    )
    xyz_label = tk.Label(
        root,
        text="XYZ Position:\n[ X:0.00, Y:0.00, Z:0.00 ]",
        font=("Courier", 12, "bold"),
        fg="red"
    )
    rad_label.pack(pady=2)
    deg_label.pack(pady=2)
    xyz_label.pack(pady=2)

    # NOTE: state list are :  TUNING , CROUCH, IDLE, WALK, RUN, JUMP
    state = tk.StringVar(value="TUNING")

    state_label = tk.Label(root, text=f"Change State {state.get()}", font=font_style)
    state_label.pack()

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
        state_label.config(text=f"Change State {state.get()}")
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
                node.walking = True

                node.walk_timer = node.create_timer(
                    node.dt, 
                    lambda: node.walk_process(kinematics)
                )

    # NOTE: 
    # ----- RADIO BUTTON ------

    tuningBtn = tk.Radiobutton(
        root,
        text="TUNING",
        value="TUNING",
        variable=state,
        font=font_style,
        command=update_state,
    )
    tuningBtn.pack()

    crouchBtn = tk.Radiobutton(
        root,
        text="CROUCH",
        value="CROUCH",
        variable=state,
        font=font_style,
        command=update_state,
    )
    crouchBtn.pack()

    idleBtn = tk.Radiobutton(
        root,
        text="IDLE",
        value="IDLE",
        variable=state,
        font=font_style,
        command=update_state,
    )
    idleBtn.pack()

    walkBtn = tk.Radiobutton(
        root,
        text="WALK",
        value="WALK",
        variable=state,
        font=font_style,
        command=update_state,
    )
    walkBtn.pack()

    phase_s = tk.StringVar(value="CROSS")
    phase_state_label = tk.Label(root, text=f"Change Phase Type {phase_s.get()}", font=font_style)
    phase_state_label.pack()

    def update_phase():
        phase_state_label.config(text=f"Change State {phase_s.get()}")

        # match state.get():
        #     case "CROSS":
        #         # node.phase_offsets = {
        #         #     'FL': 0.0,
        #         #     'BR': 0.0,
        #         #     'FR': m.pi,
        #         #     'BL': m.pi,
        #         # }
        #     case "4BEAT":
        #         node.phase_offsets = {
        #             'FL': 0.0,
        #             'BR': m.pi / 2.0,
        #             'FR': m.pi,
        #             'BL': 3.0 * m.pi / 2.0,
        #         }


    cross = tk.Radiobutton(
        root,
        text="Cross Pair",
        value="CROSS",
        variable=phase_s,
        font=font_style,
        command=update_phase,
    )
    cross.pack()

    creep = tk.Radiobutton(
        root,
        text="4-beat cycle",
        value="4BEAT",
        variable=phase_s,
        font=font_style,
        command=update_phase,
    )
    creep.pack()

    # NOTE: 
    # ----- WALK TUNING ------
    tk.Label(root, text="Gait Frequency").pack()
    gf = tk.StringVar(value=str(node.gait_freq))
    tk.Entry(root, textvariable=gf).pack()

    tk.Label(root, text="X Offset (Forward/Back Shift)").pack()
    xo = tk.StringVar(value=str(node.x_off))
    tk.Entry(root, textvariable=xo).pack()

    tk.Label(root, text="Z Offset (Ride Height)").pack()
    zo = tk.StringVar(value=str(node.z_off))
    tk.Entry(root, textvariable=zo).pack()

    tk.Label(root, text="Step Length (Stride)").pack()
    sl = tk.StringVar(value=str(node.step_len))
    tk.Entry(root, textvariable=sl).pack()

    tk.Label(root, text="Step Height (Clearance)").pack()
    sh = tk.StringVar(value=str(node.step_h))
    tk.Entry(root, textvariable=sh).pack()

    tk.Label(root, text="Self Correcting Yaw Feedback").pack()
    sc = tk.StringVar(value=str(node.sc_yaw))
    tk.Entry(root, textvariable=sc).pack()

    # tk.Label(root, text="Theta2 Center").pack()
    # t2c = tk.StringVar(value=str(node.theta2_center))
    # tk.Entry(root, textvariable=t2c).pack()
    #
    # tk.Label(root, text="Theta2 Amplitude").pack()
    # t2a = tk.StringVar(value=str(node.theta2_amplitude))
    # tk.Entry(root, textvariable=t2a).pack()
    #
    # tk.Label(root, text="Theta3 Stance").pack()
    # t3l = tk.StringVar(value=str(node.theta3_lift))
    # tk.Entry(root, textvariable=t3l).pack()
    #
    # tk.Label(root, text="Theta3 Lift").pack()
    # t3s = tk.StringVar(value=str(node.theta3_stance))
    # tk.Entry(root, textvariable=t3s).pack()

    def apply_entry(event=None):
        try:
            # node.theta2_center = float(t2c.get())
            # node.theta2_amplitude = float(t2a.get()) 
            # node.theta3_lift = float(t3l.get())
            # node.theta3_stance = float(t3s.get())
            node.gait_freq = float(gf.get())
            node.x_off = float(xo.get())
            node.z_off = float(zo.get()) 
            node.step_len = float(sl.get())
            node.step_h = float(sh.get())
            node.sc_yaw = float(sc.get())
            
        except ValueError:
            pass

    apply_btn = tk.Button(root, text="Apply", command=apply_entry)
    apply_btn.pack()


    # NOTE: 
    # ----- SLIDER FOR TUNING ------

    shoulder_slider_label = tk.Label(root, text="Shoulder Angle", font=font_style)
    shoulder_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    thigh_slider_label = tk.Label(root, text="Thigh Angle", font=font_style)
    thigh_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    leg_slider_label = tk.Label(root, text="Knee Angle", font=font_style)
    leg_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
