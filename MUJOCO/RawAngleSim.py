import os
import sys
import time
import queue
import threading
import contextlib
import numpy as np
import mujoco
import mujoco.viewer
import imageio
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
sys.stdout.reconfigure(line_buffering=True)

from HARDWARE.GaitPlayback import (
    GaitPlayer, GAIT, LEG_NAMES, URDF_INIT_RAD, MOTOR_SIGN, MOTOR_OFFSET, urdf_to_motor, step_toward,
    RATE_HZ, TIME_SCALE, START_SPEED, GAIT_SPEED_MAX, STRIDE_RAMP_S, START_TOL_RAD, START_SETTLE_S,
)
from HARDWARE.LegDriver import SPEED_DEFAULT, ACCEL_DEFAULT
from LOGIC.GaitLogic import JOINT_NAMES as MJ_JOINT_NAMES      # leg -> MuJoCo joint names, same as MujocoSim
from ROS.BaseGUI import GUI

# RobStride 02 firmware values (RS02 manual, 4.1.14 parameter table and motor spec)
LOC_KP = 40.0              # loc_kp 0x701E default: position loop, (rad/s) per rad
SPD_KP = 6.0               # spd_kp 0x701F default: speed loop, A per (rad/s)
SPD_KI = 0.02              # spd_ki 0x7020 default
LIMIT_CUR = 23.0           # A, maximum load phase current (peak)
TORQUE_CONST = 1.22        # N*m per A
LIMIT_TORQUE = 17.0        # N*m, peak load torque

MOTOR_DT = 0.0005          # s, physics step = motor loop step (the 2 ms MuJoCo default is too coarse for the speed loop)
RUN_T = None               # s of sim time, None = until the viewer is closed
PRINT_S = 0.5              # s of sim time between status prints
ENABLE_HOLD_S = 1.0        # s holding the rest pose after Enable (the robot settles on the floor) before Stand starts
SCRIPT = None              # [(sim time s, "WALK" / "IDLE"), ...] state requests on the sim clock, same as MujocoSim; None = GUI only
RECORD_HZ = 60
VIDEO_FILE = "raw_angle_sim.mp4"

class RobStridePP:
    """PP position mode for all 12 joints, motor frame (rad at the output shaft, same numbers LegDriver sends).
    loc_ref -> trapezoid plan (vel_max, acc_set) -> position loop -> speed loop -> current -> torque."""

    def __init__(self, q_motor, vel_max=SPEED_DEFAULT, acc_set=ACCEL_DEFAULT):
        self.vel_max = vel_max
        self.acc_set = acc_set
        self.loc_ref = q_motor.copy()          # Enable: first target = measured angle
        self.p_plan = q_motor.copy()
        self.v_plan = np.zeros_like(q_motor)
        self.spd_err_sum = np.zeros_like(q_motor)

    def set_target(self, loc_ref):
        self.loc_ref = np.array(loc_ref, dtype=float)

    def torque(self, q_motor, qd_motor, dt):
        # trapezoid plan: speed toward the target at most vel_max, change at most acc_set, brake in time to stop on it
        err = self.loc_ref - self.p_plan
        v_goal = np.sign(err) * np.minimum(self.vel_max, np.sqrt(2.0 * self.acc_set * np.abs(err)))
        self.v_plan = step_toward(self.v_plan, v_goal, self.acc_set * dt)
        self.p_plan += self.v_plan * dt
        arrived = np.abs(self.loc_ref - self.p_plan) < self.acc_set * dt * dt
        self.p_plan[arrived] = self.loc_ref[arrived]
        self.v_plan[arrived] = 0.0

        spd_ref = LOC_KP * (self.p_plan - q_motor)
        spd_err = spd_ref - qd_motor
        self.spd_err_sum += spd_err * dt
        current = np.clip(SPD_KP * spd_err + SPD_KI * self.spd_err_sum, -LIMIT_CUR, LIMIT_CUR)
        return np.clip(TORQUE_CONST * current, -LIMIT_TORQUE, LIMIT_TORQUE)


def to_motor(q_urdf):
    # 12 URDF angles (LEG_NAMES order) -> 12 motor angles
    return np.concatenate([urdf_to_motor(leg, q_urdf[3 * i: 3 * i + 3]) for i, leg in enumerate(LEG_NAMES)])


def main(headless=False, run_t=RUN_T, script=SCRIPT):
    # ---------------- 1. Model ----------------
    script_dir = Path(__file__).resolve().parent
    scene_path = str(script_dir.parent / 'urdf' / 'scene.xml')
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = MOTOR_DT
    data = mujoco.MjData(model)

    joint_names = [j for leg in LEG_NAMES for j in MJ_JOINT_NAMES[leg]]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_names]
    qpos_idx = model.jnt_qposadr[jids]
    qvel_idx = model.jnt_dofadr[jids]
    act_idx = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_motor") for j in joint_names])
    sign = np.concatenate([MOTOR_SIGN[leg] for leg in LEG_NAMES])      # motor = sign * q_urdf + offset
    offset = np.concatenate([MOTOR_OFFSET[leg] for leg in LEG_NAMES])

    # start like the real robot: lying in the rest pose (motor 0 = start-up zero), resting on the floor
    q_rest = (np.zeros(12) - offset) / sign
    data.qpos[qpos_idx] = q_rest
    mujoco.mj_forward(model, data)
    while data.ncon > 0:
        data.qpos[2] += 0.002
        mujoco.mj_forward(model, data)

    # ---------------- 2. Gait + IK (same object the hardware uses) and motors ----------------
    player = GaitPlayer()
    motor_stance = to_motor(np.concatenate([URDF_INIT_RAD[leg] for leg in LEG_NAMES]))
    motor_walk = to_motor(np.concatenate([player.walk_rad[leg] for leg in LEG_NAMES]))
    motors = RobStridePP(sign * data.qpos[qpos_idx] + offset)

    period = 1.0 / RATE_HZ
    tick_dec = int(round(period / MOTOR_DT))           # physics steps per 100 Hz command frame
    print_dec = int(round(PRINT_S / MOTOR_DT))
    status = {"state": "IDLE", "t_gait": 0.0, "rejects": 0, "limited": 0}
    requests = queue.Queue()                           # state names from the GUI thread / script, read on the sim clock

    def measured_motor():
        return sign * data.qpos[qpos_idx] + offset

    def latest_request():
        # newest request since the last check, None if nothing was asked
        name = None
        while not requests.empty():
            name = requests.get_nowait()
        return name

    def sequence():
        # one yield = one 100 Hz frame, same order as GaitPlayer._run (GAIT mode): stand -> walk pose -> play
        cmd = measured_motor()

        def ramp_to(goal, state):
            nonlocal cmd
            status["state"] = state
            ramp_from = cmd.copy()
            ticks = max(1, int(np.ceil(np.max(np.abs(goal - ramp_from)) / (START_SPEED * period))))
            for tick in range(1, ticks + 1):
                cmd = ramp_from + (goal - ramp_from) * (tick / ticks)
                yield cmd
            cmd = goal.copy()
            yield cmd

        status["state"] = "ENABLED"
        for _ in range(int(ENABLE_HOLD_S / period)):
            yield cmd
        yield from ramp_to(motor_stance, "MOVING_TO_START")
        for _ in range(int(START_SETTLE_S / period)):
            if np.all(np.abs(measured_motor() - motor_stance) < START_TOL_RAD):
                break
            yield cmd
        far = np.abs(measured_motor() - motor_stance) >= START_TOL_RAD
        if np.any(far):
            status["state"] = "HOLDING: stand pose not reached by " + ", ".join(np.array(joint_names)[far])
            while True:
                yield cmd

        while True:
            # IDLE: hold the stand pose until WALK is pressed
            status["state"] = "IDLE"
            while latest_request() != "WALK":
                yield cmd
            yield from ramp_to(motor_walk, "LOWERING_TO_WALK")

            # WALK: play the gait until IDLE is pressed (same as STOP GAIT on the hardware)
            status["state"] = "PLAYING"
            t_gait = 0.0
            q_sim = {leg: player.walk_rad[leg].copy() for leg in LEG_NAMES}
            while latest_request() != "IDLE":
                stride_scale = min(t_gait / STRIDE_RAMP_S, 1.0)
                limited = False
                for i, leg in enumerate(LEG_NAMES):
                    q_sim[leg], ok = player.joint_angles(leg, t_gait, q_sim[leg], stride_scale)
                    desired = urdf_to_motor(leg, q_sim[leg])
                    new_cmd = step_toward(cmd[3 * i: 3 * i + 3], desired, GAIT_SPEED_MAX * period)
                    limited = limited or not np.array_equal(new_cmd, desired)
                    status["rejects"] += int(not ok)
                    cmd[3 * i: 3 * i + 3] = new_cmd
                status["limited"] += int(limited)      # ticks with any joint at the speed cap, as GaitPlayer counts them
                status["t_gait"] = t_gait
                yield cmd
                t_gait += period * TIME_SCALE

            # back to the stand pose, then IDLE again
            yield from ramp_to(motor_stance, "RETURNING")

    frames = sequence()

    def start_gui():
        # BaseGUI as in MujocoSim, only the state buttons are used (no graph)
        gui = GUI({"state": requests.put})
        gui.setup()
        while True:
            gui.update()
            time.sleep(0.05)

    if not headless:
        threading.Thread(target=start_gui, daemon=True).start()
    print(f"[init] gait {GAIT}, time scale {TIME_SCALE}, PP vel_max {motors.vel_max} rad/s, acc_set {motors.acc_set} rad/s^2, "
          f"physics {1.0 / MOTOR_DT:.0f} Hz, commands {RATE_HZ:.0f} Hz", flush=True)

    # ---------------- 3. Loop ----------------
    step = 0
    sent = 0
    x0 = float(data.qpos[0])
    tau_peak = np.zeros(12)
    sat_steps = 0
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    last_render = time.time()
    next_frame_t = 0.0
    with (contextlib.nullcontext() if headless else mujoco.Renderer(model, height=480, width=640)) as renderer, \
         (contextlib.nullcontext() if headless else imageio.get_writer(VIDEO_FILE, fps=RECORD_HZ)) as video_writer, \
         (contextlib.nullcontext() if headless else mujoco.viewer.launch_passive(model, data)) as viewer:
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
        while headless or viewer.is_running():  # type: ignore
            step_start = time.time()
            if step % tick_dec == 0:
                while script and sent < len(script) and data.time >= script[sent][0]:
                    requests.put(script[sent][1])
                    sent += 1
                motors.set_target(next(frames))

            q_motor = measured_motor()
            qd_motor = sign * data.qvel[qvel_idx]
            tau_motor = motors.torque(q_motor, qd_motor, MOTOR_DT)
            data.ctrl[act_idx] = sign * tau_motor              # motor torque -> URDF joint torque
            tau_peak = np.maximum(tau_peak, np.abs(tau_motor))
            sat_steps += int(np.any(np.abs(tau_motor) >= LIMIT_TORQUE))

            if step % print_dec == 0:
                R = data.xmat[base_id].reshape(3, 3)
                roll = np.arctan2(R[2, 1], R[2, 2])
                pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))     # + = nose down
                print(f"t={data.time:6.2f} {status['state']:<16} t_gait={status['t_gait']:6.2f} "
                      f"x={data.qpos[0] - x0:+.3f} z={data.qpos[2]:.3f} roll={roll:+.3f} pitch={pitch:+.3f} "
                      f"|tau|max={np.max(tau_peak):5.2f} sat={sat_steps} ik_rej={status['rejects']} "
                      f"spd_lim={status['limited']}", flush=True)
                tau_peak[:] = 0.0
                sat_steps = 0

            mujoco.mj_step(model, data)
            step += 1
            if run_t is not None and data.time >= run_t:
                break
            if not headless:
                if time.time() - last_render > 0.016:
                    viewer.sync()  # type: ignore
                    last_render = time.time()
                if data.time >= next_frame_t:
                    mujoco.mjv_updateScene(model, data, viewer.opt, None, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)  # type: ignore
                    video_writer.append_data(renderer.render())  # type: ignore
                    next_frame_t += 1.0 / RECORD_HZ
                elapsed = time.time() - step_start
                if elapsed < MOTOR_DT:
                    time.sleep(MOTOR_DT - elapsed)


if __name__ == "__main__":
    main()
