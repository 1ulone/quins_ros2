from operator import ge
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
from std_msgs.msg import String, Float64MultiArray

class Gui(Node):
    def __init__(self):
        super().__init__('quins_tuner_gui')
        self.state_pub = self.create_publisher(String, '/tuner/state', 10)
        self.params_pub = self.create_publisher(Float64MultiArray, '/tuner/params', 10)
        self.phase_offsets_pub = self.create_publisher(Float64MultiArray, '/tuner/phase_offsets', 10)
        self.raw_pub = self.create_publisher(Float64MultiArray, '/tuner/raw', 10)
        self.create_subscription(Float64MultiArray, '/tuner/graph', self.graph_callback, 10)
        self.time_history = deque(maxlen=200)
        self.desired_history = deque(maxlen=200)
        self.measured_history = deque(maxlen=200)

    def graph_callback(self, msg: Float64MultiArray):
        self.time_history.append(msg.data[0])
        self.desired_history.append(msg.data[1])
        self.measured_history.append(msg.data[2])

def main(args=None):
    rclpy.init(args=args)
    node = Gui()

    def lerp(a, b, t):
        return a + t * (b - a)

    def logic_process(s, t, k):
        raw_msg = Float64MultiArray()
        raw_msg.data = [s, t, k] 
        node.raw_pub.publish(raw_msg)

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
        if len(node.time_history) > 1:
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
    state_msg = String()
    state_msg.data = state.get()
    node.state_pub.publish(state_msg)

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
        state_msg = String()
        state_msg.data = state.get()
        node.state_pub.publish(state_msg)

        if state.get() == "TUNING":
            on_tuning(True)
        else:
            on_tuning(False)

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
        phase_msg = Float64MultiArray()

        match phase_s.get():
            case "CROSS":
                phase_offsets = {
                    'FL': 0.0,
                    'BR': 0.0,
                    'FR': m.pi,
                    'BL': m.pi,
                }
                phase_msg.data = list(phase_offsets.values())
                node.phase_offsets_pub.publish(phase_msg)
            case "4BEAT":
                phase_offsets = {
                    'FL': 0.0,
                    'BR': m.pi / 2.0,
                    'FR': m.pi,
                    'BL': 3.0 * m.pi / 2.0,
                }
                phase_msg.data = list(phase_offsets.values())
                node.phase_offsets_pub.publish(phase_msg)

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
    gf = tk.StringVar(value=str(1.0))
    tk.Entry(gf_frame, textvariable=gf).pack()

    xo_frame = ttk.Frame(wt_group)
    xo_frame.grid(row=0, column=1)
    tk.Label(xo_frame, text="X Offset (Forward/Back Shift)").pack()
    xo = tk.StringVar(value=str(0.25))
    tk.Entry(xo_frame, textvariable=xo).pack()

    zo_frame = ttk.Frame(wt_group)
    zo_frame.grid(row=0, column=2)
    tk.Label(zo_frame, text="Z Offset (Ride Height)").pack()
    zo = tk.StringVar(value=str(2.7))
    tk.Entry(zo_frame, textvariable=zo).pack()

    sl_frame = ttk.Frame(wt_group)
    sl_frame.grid(row=1, column=0)
    tk.Label(sl_frame, text="Step Length (Stride)").pack()
    sl = tk.StringVar(value=str(2.0))
    tk.Entry(sl_frame, textvariable=sl).pack()

    sh_frame = ttk.Frame(wt_group)
    sh_frame.grid(row=1, column=1)
    tk.Label(sh_frame, text="Step Height (Clearance)").pack()
    sh = tk.StringVar(value=str(0.75))
    tk.Entry(sh_frame, textvariable=sh).pack()

    sc_frame = ttk.Frame(wt_group)
    sc_frame.grid(row=1, column=2)
    tk.Label(sc_frame, text="Self Correcting Yaw Feedback").pack()
    sc = tk.StringVar(value=str(0.9))
    tk.Entry(sc_frame, textvariable=sc).pack()

    param_data = [
        float(gf.get()),
        float(xo.get()),
        float(zo.get()),
        float(sl.get()),
        float(sh.get()),
        float(sc.get()),
    ]
    param_msg = Float64MultiArray()
    param_msg.data = param_data
    node.params_pub.publish(param_msg)

    def apply_entry(event=None):
        try:
            param_data = [
                float(gf.get()),
                float(xo.get()),
                float(zo.get()),
                float(sl.get()),
                float(sh.get()),
                float(sc.get()),
            ]
            param_msg = Float64MultiArray()
            param_msg.data = param_data
            node.params_pub.publish(param_msg)
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
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    root.mainloop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
