import os
import sys
import time
import glob
import tkinter as tk
from tkinter import ttk
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from HARDWARE.ATSerialBus import ATSerialBus
from HARDWARE.LegDriver import LegDriver, ENABLED, FRESH_S, SPEED_DEFAULT, ACCEL_DEFAULT
from HARDWARE.GaitPlayback import GaitPlayer, LEG_SETUP, JOINT_NAMES, GAIT, TIME_SCALE, TILT_ADJACENT_DROP_M, TILT_DIAGONAL_EXTRA_M
from HARDWARE import RobstrideProtocol as rp

ACTIVE_LEGS = ("FL", "FR", "BL", "BR")            # legs plugged in right now
JOG_STEP_RAD = 0.175
REFRESH_MS = 100
PORT_PROBE_WAIT_S = 0.1          # s, how long each port gets to answer the probe


def find_leg_ports(legs):
    # which serial port has which leg: send a stop frame to every motor ID on every /dev/ttyUSB* port
    # (motors stay off, each one on that bus replies), then match the IDs that answered to LEG_SETUP
    ports = sorted(glob.glob("/dev/ttyUSB*"))
    leg_ports, replies_per_port = {}, {}
    for port in ports:
        replied_ids = set()

        def note_reply(can_id, data, replied_ids=replied_ids):
            feedback = rp.unpack_feedback(can_id, data)
            if feedback is not None:
                replied_ids.add(feedback["motor_id"])

        bus = ATSerialBus(port)
        bus.set_handler(note_reply)
        for leg in legs:
            for motor_id in LEG_SETUP[leg][1]:
                bus.send_frame(*rp.pack_stop(motor_id))
        time.sleep(PORT_PROBE_WAIT_S)
        bus.close()
        replies_per_port[port] = sorted(replied_ids)
        for leg in legs:
            if set(LEG_SETUP[leg][1]) <= replied_ids:
                leg_ports[leg] = port
    missing = [leg for leg in legs if leg not in leg_ports]
    if missing:
        raise RuntimeError(f"no port has all three motors of {missing}; motor IDs that answered per port: {replies_per_port}")
    return leg_ports


def main():
    leg_ports = find_leg_ports(ACTIVE_LEGS)      # found by asking each port
    print(f"[gui] leg ports: {leg_ports}", flush=True)
    buses = {leg: ATSerialBus(leg_ports[leg]) for leg in ACTIVE_LEGS}
    drivers = {leg: LegDriver(buses[leg], list(JOINT_NAMES), LEG_SETUP[leg][1]) for leg in ACTIVE_LEGS}
    # the start-up position is always the rest pose: every motor's current angle becomes its zero
    for driver in drivers.values():
        for j in range(len(JOINT_NAMES)):
            driver.set_zero(j)
    gait = GaitPlayer(drivers)

    def enable_all():
        for driver in drivers.values():
            driver.enable()

    def estop_all():
        for driver in drivers.values():
            driver.estop()

    root = tk.Tk()
    root.title("Leg Joint Control (position mode)")

    # ---- Top bar: leg states, Enable, E-STOP, STOP GAIT (all buttons act on every active leg) ----
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")
    state_label = ttk.Label(top, text="Legs: -", font=("TkDefaultFont", 11, "bold"))
    state_label.pack(side="left")
    tk.Button(top, text="Enable", width=10, command=enable_all).pack(side="left", padx=15)
    tk.Button(top, text="EMERGENCY STOP", bg="red", fg="white", activebackground="darkred",
              activeforeground="white", font=("TkDefaultFont", 14, "bold"), command=estop_all).pack(side="right")
    tk.Button(top, text="STOP GAIT (to stance)", bg="orange", fg="black", activebackground="darkorange",
              font=("TkDefaultFont", 14, "bold"), command=gait.stop).pack(side="right", padx=10)

    # ---- Speed / acceleration for all joints (applied while stopped, used at the next Enable) ----
    limits = ttk.Frame(root, padding=(10, 0))
    limits.pack(fill="x")
    speed_var = tk.StringVar(value=str(SPEED_DEFAULT))
    accel_var = tk.StringVar(value=str(ACCEL_DEFAULT))
    ttk.Label(limits, text="speed").pack(side="left")
    ttk.Entry(limits, textvariable=speed_var, width=7).pack(side="left", padx=(4, 2))
    ttk.Label(limits, text="rad/s").pack(side="left", padx=(0, 12))
    ttk.Label(limits, text="accel").pack(side="left")
    ttk.Entry(limits, textvariable=accel_var, width=7).pack(side="left", padx=(4, 2))
    ttk.Label(limits, text="rad/s^2").pack(side="left", padx=(0, 12))

    # ---- Gait playback ----
    gait_row = ttk.Frame(root, padding=(10, 6, 10, 0))
    gait_row.pack(fill="x")
    ttk.Label(gait_row, text=f"gait {GAIT} ({', '.join(ACTIVE_LEGS)})   time scale").pack(side="left")
    scale_var = tk.StringVar(value=str(TIME_SCALE))
    ttk.Entry(gait_row, textvariable=scale_var, width=6).pack(side="left", padx=(4, 12))
    gait_label = ttk.Label(root, text="", padding=(10, 2, 10, 0))     # gait state / message
    gait_label.pack(fill="x")

    message_label = ttk.Label(root, text="", padding=(10, 6, 10, 0))   # last driver event per leg
    message_label.pack(fill="x")
    input_label = ttk.Label(root, text="", foreground="red", padding=(10, 0))   # typing errors
    input_label.pack(fill="x")

    def apply_limits():
        try:
            speed, accel = float(speed_var.get()), float(accel_var.get())
        except ValueError:
            input_label.config(text="speed / accel: not a number")
            return
        input_label.config(text="")
        for driver in drivers.values():
            driver.set_motion_limits(speed, accel)

    def start_gait():
        try:
            time_scale = float(scale_var.get())
        except ValueError:
            input_label.config(text="time scale: not a number")
            return
        input_label.config(text="")
        gait.start(time_scale)

    ttk.Button(limits, text="Apply", command=apply_limits).pack(side="left")
    ttk.Button(gait_row, text="Start gait", command=start_gait).pack(side="left")
    ttk.Button(gait_row, text="Stand", command=gait.stand).pack(side="left", padx=6)    # all legs to the stance, then hold
    ttk.Button(gait_row, text="Rest", command=gait.rest).pack(side="left")              # all legs to the rest pose, then motors off
    # ---- Tilt drops (m) for the tilt gait: neighbour legs, and the extra for the diagonal leg ----
    tilt_row = ttk.Frame(root, padding=(10, 6, 10, 0))
    tilt_row.pack(fill="x", after=gait_row)
    adjacent_var = tk.StringVar(value=str(TILT_ADJACENT_DROP_M * 1000.0))
    diagonal_var = tk.StringVar(value=str(TILT_DIAGONAL_EXTRA_M * 1000.0))
    ttk.Label(tilt_row, text="tilt: neighbour drop").pack(side="left")
    ttk.Entry(tilt_row, textvariable=adjacent_var, width=6).pack(side="left", padx=(4, 2))
    ttk.Label(tilt_row, text="mm   diagonal extra").pack(side="left")
    ttk.Entry(tilt_row, textvariable=diagonal_var, width=6).pack(side="left", padx=(4, 2))
    ttk.Label(tilt_row, text="mm").pack(side="left", padx=(0, 12))

    def apply_tilt():
        try:
            adjacent_m, diagonal_extra_m = float(adjacent_var.get()) / 1000.0, float(diagonal_var.get()) / 1000.0   # mm in the GUI, m inside
        except ValueError:
            input_label.config(text="tilt drops: not a number")
            return
        input_label.config(text="")
        gait.set_tilt(adjacent_m, diagonal_extra_m)

    ttk.Button(tilt_row, text="Apply tilt", command=apply_tilt).pack(side="left")

    # ---- One column per leg, one box per joint ----
    joints_frame = ttk.Frame(root, padding=10)
    joints_frame.pack()
    joint_labels = {}      # (leg, joint index): (measured, target, info)

    def send_angle(leg, j, text_var):
        try:
            angle_rad = float(text_var.get())
        except ValueError:
            input_label.config(text=f"{leg} {JOINT_NAMES[j]}: '{text_var.get()}' is not a number")
            return
        input_label.config(text="")
        drivers[leg].set_target_rad(j, angle_rad)

    for column, leg in enumerate(ACTIVE_LEGS):
        for j, name in enumerate(JOINT_NAMES):
            box = ttk.LabelFrame(joints_frame, text=f"{leg} {name} (motor {LEG_SETUP[leg][1][j]})", padding=8)
            box.grid(row=j, column=column, padx=5, pady=5, sticky="n")

            measured = ttk.Label(box, text="measured: -", width=34)
            measured.pack(anchor="w")
            target = ttk.Label(box, text="target: -")
            target.pack(anchor="w")
            info = ttk.Label(box, text="")
            info.pack(anchor="w")

            # gait angles for this joint: start pose, range and peak speed over one cycle, plus the live value
            lo, hi = gait.cycle_motor[leg][:, j].min(), gait.cycle_motor[leg][:, j].max()
            # ttk.Label(box, text=f"gait start: motor {gait.motor_start[leg][j]:+.3f} rad").pack(anchor="w", pady=(8, 0))
            # ttk.Label(box, text=f"gait range: motor {lo:+.3f} .. {hi:+.3f} rad").pack(anchor="w")
            # ttk.Label(box, text=f"gait peak speed: {gait.cycle_peak_speed[leg][j]:.2f} rad/s x time scale").pack(anchor="w")
            # gait_now = ttk.Label(box, text="gait now: -")
            # gait_now.pack(anchor="w")
            # joint_labels[(leg, j)] = (measured, target, info, gait_now)
            joint_labels[(leg, j)] = (measured, target, info)

            # absolute angle (motor rad): type it, press Send (or Enter)
            angle_row = ttk.Frame(box)
            angle_row.pack(fill="x", pady=(8, 0))
            angle_var = tk.StringVar(value="0.0")
            entry = ttk.Entry(angle_row, textvariable=angle_var, width=8)
            entry.pack(side="left")
            entry.bind("<Return>", lambda event, leg=leg, j=j, v=angle_var: send_angle(leg, j, v))
            ttk.Label(angle_row, text="rad").pack(side="left", padx=(2, 6))
            ttk.Button(angle_row, text="Send", command=lambda leg=leg, j=j, v=angle_var: send_angle(leg, j, v)).pack(side="left")

            # jog: move the target by a fixed step
            jog_row = ttk.Frame(box)
            jog_row.pack(pady=(8, 0))
            ttk.Button(jog_row, text=f"- {JOG_STEP_RAD:.3f} rad",
                       command=lambda leg=leg, j=j: drivers[leg].jog_rad(j, -JOG_STEP_RAD)).pack(side="left", padx=3)
            ttk.Button(jog_row, text=f"+ {JOG_STEP_RAD:.3f} rad",
                       command=lambda leg=leg, j=j: drivers[leg].jog_rad(j, JOG_STEP_RAD)).pack(side="left", padx=3)

            # set zero: this joint's current angle becomes 0 (only while stopped)
            ttk.Button(box, text="Set Zero (current angle = 0)",
                       command=lambda leg=leg, j=j: drivers[leg].set_zero(j)).pack(fill="x", pady=(8, 0))

    # ---- Label refresh from the drivers and the gait player (runs on the tkinter thread only) ----
    def refresh():
        statuses = {leg: drivers[leg].status() for leg in ACTIVE_LEGS}
        gs = gait.status()
        first = statuses[ACTIVE_LEGS[0]]
        state_label.config(text="   ".join(f"{leg}: {st['leg_state']}" for leg, st in statuses.items())
                                + f"   (speed {first['speed']} rad/s, accel {first['accel']} rad/s^2)")
        message_label.config(text="   |   ".join(f"{leg}: {st['message']}" for leg, st in statuses.items()))
        gait_label.config(text=f"gait: {gs['state']}   t {gs['t_gait']:.2f} s   IK rejects {gs['ik_rejects']}   "
                               f"speed-limited ticks {gs['speed_limited_ticks']}   {gs['message']}")
        now = time.perf_counter()
        for (leg, j), (measured, target, info) in joint_labels.items():
            st = statuses[leg]
            age = now - st["rx_time"][j]
            if np.isfinite(age) and age < FRESH_S:
                measured.config(text=f"measured: {st['q'][j]:+.3f} rad")
            else:
                measured.config(text="measured: NO REPLY")
            if st["leg_state"] == ENABLED:
                target.config(text=f"target:   {st['target'][j]:+.3f} rad")
            else:
                target.config(text="target:   (not enabled)")
            info.config(text=f"state {st['mode'][j]}   run_mode {st['run_mode'][j]}   "
                             f"fault {st['fault_bits'][j]:#x}   {st['temp'][j]:.1f} C")
            # gait_now.config(text=f"gait now: sim {gs['q_sim'][leg][j]:+.3f}  motor {gs['motor_target'][leg][j]:+.3f} rad")
        root.after(REFRESH_MS, refresh)

    def on_close():
        gait.stop()             # stop the gait clock first
        estop_all()             # every leg starts braking at the same moment
        for leg in ACTIVE_LEGS:
            drivers[leg].close()
            buses[leg].close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_close()


if __name__ == "__main__":
    main()
