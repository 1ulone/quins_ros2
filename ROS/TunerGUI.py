import time
import rclpy
import threading

import math as m 
import tkinter as tk
from tkinter import ttk

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from collections import deque
from tkinter.font import Font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from std_msgs.msg import String, Float64MultiArray, Int8MultiArray

class Gui(Node):
    def __init__(self):
        super().__init__('quins_tuner_gui')
        self.state_pub = self.create_publisher(String, '/tuner/state', 10)
        self.params_pub = self.create_publisher(Float64MultiArray, '/tuner/params', 10)
        self.jparams_pub = self.create_publisher(Float64MultiArray, '/tuner/jparams', 10)
        self.phase_offsets_pub = self.create_publisher(Float64MultiArray, '/tuner/phase_offsets', 10)
        self.raw_pub = self.create_publisher(Float64MultiArray, '/tuner/raw', 10)
        self.create_subscription(Float64MultiArray, '/tuner/graph', self.graph_callback, 10)
        self.create_subscription(Int8MultiArray, '/tuner/contacts', self.foot_contacts_callback, 10)
        self.time_history = deque(maxlen=200)
        self.desired_history = deque(maxlen=200)
        self.measured_history = deque(maxlen=200)

        self.foot_contacts = {
            'FL': False,
            'BR': False,
            'FR': False,
            'BL': False,
        }

    def graph_callback(self, msg: Float64MultiArray):
        self.time_history.append(msg.data[0])
        self.desired_history.append(msg.data[1])
        self.measured_history.append(msg.data[2])

    def foot_contacts_callback(self, msg: Int8MultiArray):
        self.foot_contacts = {
            'FL': True if msg.data[0]==1 else False,
            'BR': True if msg.data[1]==1 else False,
            'FR': True if msg.data[2]==1 else False,
            'BL': True if msg.data[3]==1 else False,
        }

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
    fig = Figure(figsize=(5.0, 2.5), dpi=100)
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
            slider_group.grid(row=4, column=0)
            shoulder_slider.pack()
            shoulder_slider_label.pack()
            thigh_slider.pack()
            thigh_slider_label.pack()
            leg_slider.pack()
            leg_slider_label.pack()
        else:
            slider_group.grid_forget()
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

        if state.get() == "JUMP":
            node.time_history.clear()
            node.desired_history.clear()
            node.measured_history.clear()

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

    jumpBtn = tk.Radiobutton(
        state_group,
        text="JUMP",
        value="JUMP",
        variable=state,
        font=font_style,
        command=update_state,
    )
    jumpBtn.grid(column=0, row=2)

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

    jt_group = ttk.LabelFrame(grid_container, text="JUMP TUNING PARAMS", padding=15)
    jt_group.grid(row=3, column=0, columnspan=2)

    yc_frame = ttk.Frame(jt_group)
    yc_frame.grid(row=0, column=0)
    tk.Label(yc_frame, text="Y Crouch").pack()
    yc = tk.StringVar(value=str(0.75))
    tk.Entry(yc_frame, textvariable=yc).pack()

    yt_frame = ttk.Frame(jt_group)
    yt_frame.grid(row=0, column=1)
    tk.Label(yt_frame, text="Y Thrust").pack()
    yt = tk.StringVar(value=str(5.0))
    tk.Entry(yt_frame, textvariable=yt).pack()

    yf_frame = ttk.Frame(jt_group)
    yf_frame.grid(row=0, column=2)
    tk.Label(yf_frame, text="Y Flight").pack()
    yf = tk.StringVar(value=str(1.0))
    tk.Entry(yf_frame, textvariable=yf).pack()

    xt_frame = ttk.Frame(jt_group)
    xt_frame.grid(row=1, column=0)
    tk.Label(xt_frame, text="X Thrust").pack()
    xt = tk.StringVar(value=str(0.6))
    tk.Entry(xt_frame, textvariable=xt).pack()

    xf_frame = ttk.Frame(jt_group)
    xf_frame.grid(row=1, column=1)
    tk.Label(xf_frame, text="X Flight").pack()
    xf = tk.StringVar(value=str(1.0))
    tk.Entry(xf_frame, textvariable=xf).pack()

    xc_frame = ttk.Frame(jt_group)
    xc_frame.grid(row=1, column=2)
    tk.Label(xc_frame, text="X Catch").pack()
    xc = tk.StringVar(value=str(-1))
    tk.Entry(xc_frame, textvariable=xc).pack()

    pt_frame = ttk.Frame(jt_group)
    pt_frame.grid(row=2, column=0)
    tk.Label(pt_frame, text="Prepare Time").pack()
    pt = tk.StringVar(value=str(1.5))
    tk.Entry(pt_frame, textvariable=pt).pack()

    tt_frame = ttk.Frame(jt_group)
    tt_frame.grid(row=2, column=1)
    tk.Label(tt_frame, text="Thrust Time").pack()
    tt = tk.StringVar(value=str(0.5))
    tk.Entry(tt_frame, textvariable=tt).pack()

    ft_frame = ttk.Frame(jt_group)
    ft_frame.grid(row=2, column=2)
    tk.Label(ft_frame, text="Flight Time").pack()
    ft = tk.StringVar(value=str(0.15))
    tk.Entry(ft_frame, textvariable=ft).pack()

    lt_frame = ttk.Frame(jt_group)
    lt_frame.grid(row=3, column=0)
    tk.Label(lt_frame, text="Landing Time").pack()
    lt = tk.StringVar(value=str(0.5))
    tk.Entry(lt_frame, textvariable=lt).pack()

    ct_frame = ttk.Frame(jt_group)
    ct_frame.grid(row=3, column=1)
    tk.Label(ct_frame, text="Catch Time").pack()
    ct = tk.StringVar(value=str(0.2))
    tk.Entry(ct_frame, textvariable=ct).pack()

    jparam_data = [
        float(yc.get()),
        float(yt.get()),
        float(yf.get()),
        float(xt.get()),
        float(xf.get()),
        float(xc.get()),
        float(pt.get()),
        float(tt.get()),
        float(ft.get()),
        float(lt.get()),
        float(ct.get()),
    ]
    jparam_msg = Float64MultiArray()
    jparam_msg.data = jparam_data
    node.jparams_pub.publish(jparam_msg)

    def apply_jump_entry(event=None):
        try:
            jparam_data = [
                float(yc.get()),
                float(yt.get()),
                float(yf.get()),
                float(xt.get()),
                float(xf.get()),
                float(xc.get()),
                float(pt.get()),
                float(tt.get()),
                float(ft.get()),
                float(lt.get()),
                float(ct.get()),
            ]
            jparam_msg = Float64MultiArray()
            jparam_msg.data = jparam_data
            node.jparams_pub.publish(jparam_msg)
        except ValueError:
            pass

    j_apply_btn = tk.Button(jt_group, text="Apply JP", command=apply_jump_entry)
    j_apply_btn.grid(row=3, column=2)

    contact_group = ttk.LabelFrame(grid_container, text="Contacts Foot Boolean", padding=15)
    contact_group.grid(row=3, column=2)
    contact_labels = {
        'FL': tk.Label(contact_group, text=f"FL : {node.foot_contacts['FL']}"),
        'FR': tk.Label(contact_group, text=f"FR : {node.foot_contacts['FR']}"),
        'BL': tk.Label(contact_group, text=f"BL : {node.foot_contacts['BL']}"),
        'BR': tk.Label(contact_group, text=f"BR : {node.foot_contacts['BR']}")
    }

    for label in contact_labels.values():
        label.pack()

    def refresh_contacts():
        contact_labels['FL'].config(text=f"FL : {node.foot_contacts['FL']}")
        contact_labels['FR'].config(text=f"FR : {node.foot_contacts['FR']}")
        contact_labels['BL'].config(text=f"BL : {node.foot_contacts['BL']}")
        contact_labels['BR'].config(text=f"BR : {node.foot_contacts['BR']}")
        
        root.after(100, refresh_contacts)

    root.after(100, refresh_contacts)

    slider_group = ttk.LabelFrame(grid_container, text="Slider", padding=0)
    shoulder_slider_label = tk.Label(slider_group, text="Shoulder Angle", font=font_style)
    shoulder_slider = tk.Scale(slider_group, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    thigh_slider_label = tk.Label(slider_group, text="Thigh Angle", font=font_style)
    thigh_slider = tk.Scale(slider_group, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    leg_slider_label = tk.Label(slider_group, text="Knee Angle", font=font_style)
    leg_slider = tk.Scale(slider_group, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    root.mainloop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
