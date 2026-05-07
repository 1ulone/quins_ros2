from tkinter.font import Font
from rclpy.node import Node
from typing import Optional
import rclpy
import json
import threading
import time
import math as m 
import tkinter as tk
from builtin_interfaces.msg import Duration
from quins.KinematicsLogic import KinematicsLogic
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from quins.MqttLogic import MqttLogic
from nav_msgs.msg import Odometry

LEG_NAMES = ['FL', 'FR', 'BL', 'BR']

PHASE_OFFSETS = {
    'FL': 0.0,
    'BR': 0.0,
    'FR': m.pi,
    'BL': m.pi,
}

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
            '/model/quadruped/odometry',
            self.odometry_callback,
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

        self.mqtt = MqttLogic()
        self.timer = self.create_timer(2.0, self.send_mqtt)

        self.gait_freq = 4.0
        self.x_off = 1.2925
        self.z_off = -2.7
        self.step_len = 2.5
        self.step_h = 0.75
        self.sc_yaw = 1.25

        self.target_yaw = 0.0

        # self.theta2_center    = 5.0  # deg
        # self.theta2_amplitude = 10.0  # deg
        # self.theta3_stance    = 20.0  # deg
        # self.theta3_lift      = 25.0  # deg

        self.control_rate = 50.0
        self.dt = 1.0 / self.control_rate
        self.t = 0.0
        self.walking = False
        self.walk_thread: Optional[threading.Thread] = None

    def odometry_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation

        sin_y_cos_pos = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y_cos_pos = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self. current_yaw = m.atan2(sin_y_cos_pos, cos_y_cos_pos)

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


    def send_mqtt(self):
        self.mqtt.client.publish(self.mqtt.main_topic, json.dumps(self.phi))

    def seconds_to_duration(self, sec: float) -> Duration:
        d = Duration()
        d.sec     = int(sec)
        d.nanosec = int((sec - int(sec)) * 1e9)
        return d

    def walk_process(self, k: KinematicsLogic):
        omega = 2.0 * m.pi * self.gait_freq

        msg = JointTrajectory()
        msg.joint_names = []

        for leg in LEG_NAMES:
            msg.joint_names += JOINT_NAMES[leg]

        # make the Controller to look ahead of what steps to follow
        # instead of a single fixed target
        lookahead_steps = 5
        points = []

        ramp_duration = 1.0
        ramp_factor = min(self.t / ramp_duration, 1.0)
        base_step_len = self.step_len * ramp_factor

        yaw_error = self.target_yaw - self.current_yaw

        for i in range(lookahead_steps):
            t_ahead = self.t + i * self.dt
            phase_now = (omega * t_ahead) % (2.0 * m.pi)
            all_pos = []

            for leg in LEG_NAMES:
                leg_phase = (phase_now + PHASE_OFFSETS[leg]) % (2.0 * m.pi)

                active_step_len = base_step_len

                if leg in ['FL', 'BL']:
                    active_step_len += (yaw_error * self.sc_yaw) * ramp_factor
                elif leg in ['FR', 'BR']:
                    active_step_len -= (yaw_error * self.sc_yaw) * ramp_factor

                #get the cartesian xyz
                tx, ty, tz = k.gait_trajectory(leg_phase, self.x_off, self.z_off, active_step_len, self.step_h)

                theta1, theta2, theta3 = k.ik(tx, ty, tz)

                # theta1, theta2, theta3 = k.gait_angles(leg_phase, self.theta2_center, self.theta2_amplitude, self.theta3_lift, self.theta3_stance)

                if i == 0:
                    T = k.fk(theta1, theta2, theta3)
                    self.get_logger().info(
                        f"[{leg}] {'SWING ' if leg_phase < m.pi else 'STANCE'} | "
                        f"θ=({theta1:.1f}, {theta2:.1f}, {theta3:.1f}) | "
                        f"foot=({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})"
                    )

                    # for sending to MQTT 
                    phi_key = LEG_TO_PHI[leg]
                    self.phi[phi_key]["shoulder"] = m.radians(theta1)
                    self.phi[phi_key]["thigh"] = m.radians(theta2)
                    self.phi[phi_key]["leg"] = m.radians(theta3)

                all_pos += [m.radians(theta1), m.radians(theta2), m.radians(theta3)]

            point = JointTrajectoryPoint()
            point.positions = all_pos
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

        t04 = kinematics.fk(m.degrees(s), m.degrees(t), m.degrees(k))
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

        match state.get():
            case "TUNING":
                on_tuning(True)
                # Sync sliders to current position so they don't snap
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

                def walk_loop():
                    node.walking = True
                    while node.walking:
                        node.walk_process(kinematics)
                        time.sleep(node.dt * 2)

                node.walk_thread = threading.Thread(target=walk_loop, daemon=True)
                node.walk_thread.start()

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
