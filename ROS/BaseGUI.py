import math as m
import tkinter as tk
import pygame
from tkinter import ttk
from collections import deque
from tkinter.font import Font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI:
    def __init__(self, callbacks):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INPUT] Connected : {self.joystick.get_name()}")

            self.num_buttons = self.joystick.get_numbuttons()
            self.button_states = [False] * self.num_buttons
        else:
            print("[INPUT] No Gamepad detected")
            self.joystick = None

        self.root = tk.Tk()
        self.callbacks = callbacks

        self.time_history = deque(maxlen=200)
        self.desired_history = deque(maxlen=200)
        self.measured_history = deque(maxlen=200)

        self.state = tk.StringVar(master=self.root, value="TUNING") 
        self.phase_s = tk.StringVar(master=self.root, value="CROSS")
        self.wt_vars = {
            'gf': tk.StringVar(master=self.root, value="1.0"), 
            'xo': tk.StringVar(master=self.root, value="0.25"),
            'zo': tk.StringVar(master=self.root, value="2.7"), 
            'sl': tk.StringVar(master=self.root, value="2.0"),
            'sh': tk.StringVar(master=self.root, value="0.75"), 
            'sc': tk.StringVar(master=self.root, value="0.9")
        }

        self.ly = 0.0
        self.rx = 0.0
        
        self.fig = Figure(figsize=(5.0, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

    def setup(self):
        self.root.title("Quadruped Tuner")
        self.root.geometry("500x450") 

        font_style = Font(family="Arial", size=10, weight="bold") 

        grid_container = tk.Frame(self.root)
        grid_container.pack(pady=10, padx=10)

        self.ax.set_title("Joint Position Tracking")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Position (rad)")
        self.ax.grid(True)

        self.ly_display = tk.StringVar(master=self.root, value="Left Stick Y: 0.0")
        self.rx_display = tk.StringVar(master=self.root, value="Right Stick X: 0.0")

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.NONE, expand=False, pady=5)

        self.state = tk.StringVar(value="TUNING")
        state_group = ttk.LabelFrame(grid_container, text=f"Change State {self.state.get()}", padding=15)
        state_group.grid(row=0, column=0)

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
            if self.callbacks.get('state'):
                self.callbacks['state'](self.state.get())
            if self.state.get() == "JUMP":
                self.time_history.clear()
                self.desired_history.clear()
                self.measured_history.clear()

            if self.state.get() == "TUNING":
                on_tuning(True)
            else:
                on_tuning(False)

        tuningBtn = tk.Radiobutton(
            state_group,
            text="TUNING",
            value="TUNING",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        tuningBtn.grid(column=0, row=0)

        crouchBtn = tk.Radiobutton(
            state_group,
            text="CROUCH",
            value="CROUCH",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        crouchBtn.grid(column=1, row=0)

        idleBtn = tk.Radiobutton(
            state_group,
            text="IDLE",
            value="IDLE",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        idleBtn.grid(column=0, row=1)

        walkBtn = tk.Radiobutton(
            state_group,
            text="WALK",
            value="WALK",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        walkBtn.grid(column=1, row=1)

        jumpBtn = tk.Radiobutton(
            state_group,
            text="JUMP",
            value="JUMP",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        jumpBtn.grid(column=0, row=2)

        crawlBtn = tk.Radiobutton(
            state_group,
            text="CRAWL",
            value="CRAWL",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        crawlBtn.grid(column=1, row=2)

        trotBtn = tk.Radiobutton(
            state_group,
            text="RUN",
            value="RUN",
            variable=self.state,
            font=font_style,
            command=update_state,
        )
        trotBtn.grid(column=0, row=3)

        gpad_group = ttk.LabelFrame(grid_container, text=f"Gamepad group {self.phase_s.get()}", padding=15)
        gpad_group.grid(row=0, column=1, sticky='nsew')

        tk.Label(gpad_group, textvariable=self.ly_display).pack()
        tk.Label(gpad_group, textvariable=self.rx_display).pack()

        wt_group = ttk.LabelFrame(grid_container, text="WALK TUNING PARAMS", padding=15)
        wt_group.grid(row=1, column=0, columnspan=2)

        gf_frame = ttk.Frame(wt_group)
        gf_frame.grid(row=0, column=0)
        tk.Label(gf_frame, text="Gait Frequency").pack()
        gf = tk.StringVar(value=str(0.5))
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
        if self.callbacks.get("wt_params"):
            self.callbacks['wt_params'](param_data)

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
                if self.callbacks.get("wt_params"):
                    self.callbacks['wt_params'](param_data)
            except ValueError:
                pass

        apply_btn = tk.Button(wt_group, text="Apply", command=apply_entry)
        apply_btn.grid(row=2, column=0, columnspan=3, sticky='nsew')

        jt_group = ttk.LabelFrame(grid_container, text="JUMP TUNING PARAMS", padding=15)
        jt_group.grid(row=3, column=0, columnspan=2)

        yc_frame = ttk.Frame(jt_group)
        yc_frame.grid(row=0, column=0)
        tk.Label(yc_frame, text="Y Crouch").pack()
        yc = tk.StringVar(value=str(0.6))
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
        xt = tk.StringVar(value=str(2.5))
        tk.Entry(xt_frame, textvariable=xt).pack()

        xf_frame = ttk.Frame(jt_group)
        xf_frame.grid(row=1, column=1)
        tk.Label(xf_frame, text="X Flight").pack()
        xf = tk.StringVar(value=str(0.0))
        tk.Entry(xf_frame, textvariable=xf).pack()

        xc_frame = ttk.Frame(jt_group)
        xc_frame.grid(row=1, column=2)
        tk.Label(xc_frame, text="X Catch").pack()
        xc = tk.StringVar(value=str(-1.5))
        tk.Entry(xc_frame, textvariable=xc).pack()

        pt_frame = ttk.Frame(jt_group)
        pt_frame.grid(row=2, column=0)
        tk.Label(pt_frame, text="Prepare Time").pack()
        pt = tk.StringVar(value=str(0.85))
        tk.Entry(pt_frame, textvariable=pt).pack()

        ftt_frame = ttk.Frame(jt_group)
        ftt_frame.grid(row=2, column=1)
        tk.Label(ftt_frame, text="Front Thrust Time").pack()
        ftt = tk.StringVar(value=str(1.0))
        tk.Entry(ftt_frame, textvariable=ftt).pack()

        btt_frame = ttk.Frame(jt_group)
        btt_frame.grid(row=2, column=2)
        tk.Label(btt_frame, text="Back Thrust Time").pack()
        btt = tk.StringVar(value=str(0.5))
        tk.Entry(btt_frame, textvariable=btt).pack()

        ft_frame = ttk.Frame(jt_group)
        ft_frame.grid(row=3, column=0)
        tk.Label(ft_frame, text="Flight Time").pack()
        ft = tk.StringVar(value=str(0.15))
        tk.Entry(ft_frame, textvariable=ft).pack()

        lt_frame = ttk.Frame(jt_group)
        lt_frame.grid(row=3, column=1)
        tk.Label(lt_frame, text="Landing Time").pack()
        lt = tk.StringVar(value=str(0.5))
        tk.Entry(lt_frame, textvariable=lt).pack()

        ct_frame = ttk.Frame(jt_group)
        ct_frame.grid(row=3, column=2)
        tk.Label(ct_frame, text="Catch Time").pack()
        ct = tk.StringVar(value=str(0.1))
        tk.Entry(ct_frame, textvariable=ct).pack()

        stl_frame = ttk.Frame(jt_group)
        stl_frame.grid(row=4, column=0)
        tk.Label(stl_frame, text="Stablize leg").pack()
        stl = tk.StringVar(value=str(0.5))
        tk.Entry(stl_frame, textvariable=stl).pack()

        bt_frame = ttk.Frame(jt_group)
        bt_frame.grid(row=4, column=2)
        tk.Label(bt_frame, text="Back Thrust Specific").pack()
        bt = tk.StringVar(value=str(2.5))
        tk.Entry(bt_frame, textvariable=bt).pack()

        ptr_frame = ttk.Frame(jt_group)
        ptr_frame.grid(row=5, column=0)
        tk.Label(ptr_frame, text="Pitch Threshold").pack()
        ptr = tk.StringVar(value=str(-0.6))
        tk.Entry(ptr_frame, textvariable=ptr).pack()

        jparam_data = [
            float(yc.get()),
            float(yt.get()),
            float(yf.get()),
            float(xt.get()),
            float(xf.get()),
            float(xc.get()),
            float(pt.get()),
            float(ftt.get()),
            float(btt.get()),
            float(ft.get()),
            float(lt.get()),
            float(ct.get()),
            float(stl.get()),
            float(bt.get()),
            float(ptr.get()),
        ]
        if self.callbacks.get("jt_params"):
            self.callbacks["jt_params"](jparam_data)

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
                    float(ftt.get()),
                    float(btt.get()),
                    float(ft.get()),
                    float(lt.get()),
                    float(ct.get()),
                    float(stl.get()),
                    float(bt.get()),
                    float(ptr.get()),
                ]
                if self.callbacks.get("jt_params"):
                    self.callbacks["jt_params"](jparam_data)
            except ValueError:
                pass

        j_apply_btn = tk.Button(jt_group, text="Apply JP", command=apply_jump_entry)
        j_apply_btn.grid(row=4, column=1)

        def tuning_process(val):
            s = float(shoulder_slider.get())
            t = float(thigh_slider.get())
            k = float(leg_slider.get())

            if self.callbacks.get('raw_tune'):
                self.callbacks['raw_tune']([s, t, k])

        slider_group = ttk.LabelFrame(grid_container, text="Slider", padding=0)
        shoulder_slider_label = tk.Label(slider_group, text="Shoulder Angle", font=font_style)
        shoulder_slider = tk.Scale(slider_group, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

        thigh_slider_label = tk.Label(slider_group, text="Thigh Angle", font=font_style)
        thigh_slider = tk.Scale(slider_group, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

        leg_slider_label = tk.Label(slider_group, text="Knee Angle", font=font_style)
        leg_slider = tk.Scale(slider_group, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=tuning_process)

    def refresh_graph(self):
        return

        if len(self.time_history) > 1:
            self.ax.clear()

            t_data = list(self.time_history)
            
            self.ax.plot(t_data, list(self.desired_history), 'k--', label="Desired")
            self.ax.plot(t_data, list(self.measured_history), 'y-', label="Measured")
            
            # Restore formatting stripped by ax.clear()
            self.ax.set_title("Joint Position Tracking")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Position (rad)")
            self.ax.grid(True)
            self.ax.legend(loc="upper right")
            
            # Force the X-axis to scroll tightly with the time window
            self.ax.set_xlim(t_data[0], t_data[-1])
            
            self.canvas.draw()

    def update_graph(self, t, desired, measured):
        return 

        self.time_history.append(t)
        self.desired_history.append(desired)
        self.measured_history.append(measured)
        self.refresh_graph()

    def update(self):
        self.update_gamepad_input()
        self.root.update()

    def set_quad_state(self, new_state):
        self.state.set(new_state)
        
        if self.callbacks.get('state'):
            self.callbacks['state'](self.state.get())
        if self.state.get() == "JUMP":
            self.time_history.clear()
            self.desired_history.clear()
            self.measured_history.clear()

    def update_gamepad_input(self):
        if self.joystick is None:
            return
        
        button_mapping = {
            0: "JUMP",   # Cross / A
            1: "CROUCH", # Circle / B
            2: "IDLE",   # Square / X
            3: "WALK",   # Triangle / Y
            4: "CRAWL",  # L1
            5: "RUN"     # R1
        }

        # 1. Process discrete button events directly from the queue
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"[DEBUG] Raw Event Button {event.button} pressed")
                if event.button in button_mapping:
                    self.set_quad_state(button_mapping[event.button])

        # 2. Poll continuous axis states
        raw_ly = self.joystick.get_axis(1)
        raw_rx = self.joystick.get_axis(0)

        deadzone = 0.1
        self.ly = raw_ly if abs(raw_ly) > deadzone else 0.0
        self.rx = raw_rx if abs(raw_rx) > deadzone else 0.0

        self.ly_display.set(f"Left Stick Y: {self.ly:.2f}")
        self.rx_display.set(f"Right Stick X: {self.rx:.2f}")

        if self.callbacks.get("gamepad"):
            self.callbacks["gamepad"]([self.ly, self.rx])
