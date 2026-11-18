import time
import queue
import threading
import numpy as np
from HARDWARE import RobstrideProtocol as rp

# ---- Fixed settings ----
RATE_HZ = 100.0           # how often each joint's target angle (loc_ref) is re-sent while enabled
ANGLE_LIMIT = np.pi       # rad, mechanical limit: every target is clipped to +-3.14, Enable is refused outside it
SPEED_DEFAULT = 10.0      # rad/s, PP speed (vel_max): motor default, read back with ParamCheck.py
ACCEL_DEFAULT = 10.0      # rad/s^2, PP acceleration (acc_set): motor default, read back with ParamCheck.py
SPEED_MAX = 33.0          # rad/s, RS02 manual speed range (spd_ref / limit_spd: 0 to 33 rad/s)
BRAKE_MARGIN_S = 0.2      # extra wait after the worst-case braking time, before the motors are turned off
FRESH_S = 0.1             # a reply older than this counts as "no reply"

DEBUG = True
DEBUG_PRINT_S = 1.0

# Leg states
STOPPED = "STOPPED"       # motors off (limp); stop frames keep going out so the angles keep updating
ENABLED = "ENABLED"       # motors on in PP mode, each joint goes to its target angle
BRAKING = "BRAKING"       # e-stop pressed: vel_max = 0 (motor brakes and holds), then STOPPED


class LegDriver:
    """Per-joint angle control in RobStride position mode (PP) for the motors on one CAN bus (one leg).

    Who does what:
      - the bus reader thread (inside the bus object) parses every reply and calls _on_frame
      - the loop thread in this class is the ONLY thing that writes to the bus
      - the GUI only calls the public methods; they change a target or queue a request,
        they never send anything themselves
    """

    def __init__(self, bus, joint_names, motor_ids, rate_hz=RATE_HZ, host_id=rp.HOST_ID):
        assert len(set(motor_ids)) == len(motor_ids), "motor IDs on one bus must be unique"
        self.bus = bus
        self.names = list(joint_names)
        self.motor_ids = list(motor_ids)
        self.index_of_motor = {motor_id: i for i, motor_id in enumerate(self.motor_ids)}
        self.period = 1.0 / rate_hz
        self.host_id = host_id

        n = len(self.motor_ids)
        self.lock = threading.Lock()            # guards everything below (GUI thread, loop thread, reader thread)
        self.target = np.zeros(n)               # rad, angle each joint is sent to (used only while ENABLED)
        self.q = np.full(n, np.nan)             # rad, last measured angle (nan until the first reply)
        self.vel = np.zeros(n)                  # rad/s, last measured speed
        self.temp = np.zeros(n)                 # deg C
        self.fault_bits = np.zeros(n, dtype=int)
        self.mode = np.full(n, -1, dtype=int)   # motor state from the feedback frame: 0 reset (off), 2 run, -1 no reply yet
        self.run_mode = np.full(n, -1, dtype=int)   # run_mode read back after Enable: 1 = PP, -1 not read yet
        self.rx_time = np.full(n, np.nan)       # time.perf_counter() of each motor's last feedback frame
        self.rx_count = np.zeros(n, dtype=int)
        self.speed = SPEED_DEFAULT              # rad/s, written as vel_max at Enable
        self.accel = ACCEL_DEFAULT              # rad/s^2, written as acc_set at Enable
        self.leg_state = STOPPED
        self.brake_until = 0.0
        self.message = "stopped, press Enable"  # last event, shown in the GUI
        self.rx_type_count = {}
        self.next_debug_t = 0.0

        self.requests = queue.Queue()           # one-off actions (enable / estop), run by the loop thread in order
        self.running = True
        bus.set_handler(self._on_frame)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---------------- Public methods (called from the GUI) ----------------

    def enable(self):
        self.requests.put(("enable", None))

    def estop(self):
        self.requests.put(("estop", None))

    def set_zero(self, i):
        self.requests.put(("zero", i))

    def set_motion_limits(self, speed, accel):
        with self.lock:
            if self.leg_state != STOPPED:
                self.message = "speed / accel can only be changed while stopped (PP can't change them mid-move)"
                return
            if not (0.0 < speed <= SPEED_MAX) or not accel > 0.0:
                self.message = f"refused: speed must be above 0 and at most {SPEED_MAX} rad/s, accel above 0"
                return
            self.speed = float(speed)
            self.accel = float(accel)
            self.message = f"speed {self.speed} rad/s, accel {self.accel} rad/s^2, used at the next Enable"

    def set_target_rad(self, i, angle_rad):
        # new absolute target for joint i; ignored unless the leg is enabled
        with self.lock:
            if self.leg_state != ENABLED:
                self.message = "not enabled, target ignored"
                return
            self.target[i] = float(angle_rad) 


    def jog_rad(self, i, step_rad):
        # move joint i's target by step_deg from where the target is now; ignored unless enabled
        with self.lock:
            if self.leg_state != ENABLED:
                self.message = "not enabled, jog ignored"
                return
            self.target[i] = self.target[i] + step_rad 

    def status(self):
        # copy of everything the GUI shows
        with self.lock:
            return {
                "leg_state": self.leg_state,
                "message": self.message,
                "speed": self.speed,
                "accel": self.accel,
                "target": self.target.copy(),
                "q": self.q.copy(),
                "vel": self.vel.copy(),
                "temp": self.temp.copy(),
                "fault_bits": self.fault_bits.copy(),
                "mode": self.mode.copy(),
                "run_mode": self.run_mode.copy(),
                "rx_time": self.rx_time.copy(),
            }

    def close(self):
        # e-stop, wait for the braking to finish, stop the loop, then one last stop frame per motor
        self.estop()
        time.sleep(self._brake_time() + 3 * self.period)
        self.running = False
        self.thread.join(timeout=0.5)
        for motor_id in self.motor_ids:
            self.bus.send_frame(*rp.pack_stop(motor_id, host_id=self.host_id))

    # ---------------- Reply handling (bus reader thread) ----------------

    def _on_frame(self, can_id, data):
        comm_type = rp.split_id(can_id)[0]
        with self.lock:
            self.rx_type_count[comm_type] = self.rx_type_count.get(comm_type, 0) + 1

        fb = rp.unpack_feedback(can_id, data)
        if fb is not None:                      # type 2: angle, speed, temperature, fault bits, state
            i = self.index_of_motor.get(fb["motor_id"])
            if i is None:
                return
            with self.lock:
                self.q[i] = fb["pos"]
                self.vel[i] = fb["vel"]
                self.temp[i] = fb["temp"]
                self.fault_bits[i] = fb["fault_bits"]
                self.mode[i] = fb["mode"]
                self.rx_time[i] = time.perf_counter()
                self.rx_count[i] += 1
            return
        reply = rp.unpack_param_reply(can_id, data)
        if reply is not None and reply["index"] == rp.RUN_MODE_INDEX:   # type 17: run_mode read back after Enable
            i = self.index_of_motor.get(reply["motor_id"])
            if i is not None:
                with self.lock:
                    self.run_mode[i] = reply["uint"] & 0xFF

    # ---------------- Loop thread (the only bus writer) ----------------

    def _loop(self):
        next_t = time.perf_counter()
        while self.running:
            self._run_requests()

            with self.lock:
                if self.leg_state == BRAKING and time.perf_counter() >= self.brake_until:
                    self.leg_state = STOPPED
                    self.message = "e-stop done, motors off, press Enable"
                state = self.leg_state
                target = self.target.copy()
            if DEBUG:
                self._debug_print()

            for i, motor_id in enumerate(self.motor_ids):
                if state == ENABLED:
                    frame = rp.pack_param_write_float(motor_id, rp.LOC_REF_INDEX, target[i], self.host_id)
                elif state == BRAKING:
                    frame = rp.pack_param_write_float(motor_id, rp.VEL_MAX_INDEX, 0.0, self.host_id)   # brake and hold
                else:
                    frame = rp.pack_stop(motor_id, host_id=self.host_id)    # keeps the motor off, still replies
                self.bus.send_frame(*frame)

            # fixed rate on absolute deadlines; after a long stall, restart the clock instead of bursting
            next_t += self.period
            wait = next_t - time.perf_counter()
            if wait > 0.0:
                time.sleep(wait)
            elif wait < -self.period:
                next_t = time.perf_counter()

    def _run_requests(self):
        while True:
            try:
                action, arg = self.requests.get_nowait()
            except queue.Empty:
                return
            if action == "estop":
                self._estop()
            elif action == "enable":
                self._enable()
            elif action == "zero":
                self._zero(arg)

    def _brake_time(self):
        # worst case: braking from full speed at the set acceleration
        return self.speed / self.accel + BRAKE_MARGIN_S

    def _debug_print(self):
        # once per DEBUG_PRINT_S: per joint target / measured angle (rad) / speed / motor state / run_mode / fault,
        # plus frames received per type since the last print (2 = feedback, 17 = parameter reply)
        now = time.perf_counter()
        if now < self.next_debug_t:
            return
        self.next_debug_t = now + DEBUG_PRINT_S

        # with self.lock:
        #     joints = "  ".join(
        #         f"{name}: tgt {self.target[i]:+.3f} q {self.q[i]:+.3f} v {self.vel[i]:+.2f} st {self.mode[i]} rm {self.run_mode[i]} f {self.fault_bits[i]:#x} n {self.rx_count[i]}"
        #         for i, name in enumerate(self.names))
        #     counts, self.rx_type_count = self.rx_type_count, {}
        #     self.rx_count[:] = 0
        #     state = self.leg_state
        # print(f"[leg] {state:<8} {joints}   rx {counts}", flush=True)

    def _estop(self):
        with self.lock:
            if self.leg_state == ENABLED:
                self.leg_state = BRAKING
                self.brake_until = time.perf_counter() + self._brake_time()
                self.message = "E-STOP: braking (vel_max = 0), then motors off"

    def _zero(self, i):
        with self.lock:
            if self.leg_state != STOPPED:
                self.message = f"set zero refused, leg is {self.leg_state} (stop it first)"
                return
        motor_id = self.motor_ids[i]
        self.bus.send_frame(*rp.pack_param_write_uint8(motor_id, rp.RUN_MODE_INDEX, rp.RUN_MODE_OPERATION, self.host_id))
        self.bus.send_frame(*rp.pack_set_zero(motor_id, self.host_id))
        self.bus.send_frame(*rp.pack_param_read(motor_id, rp.RUN_MODE_INDEX, self.host_id))   # GUI shows rm 0 if the switch took
        with self.lock:
            self.message = f"{self.names[i]}: run_mode 0 + set zero sent, measured should read about 0"
        if DEBUG:
            print(f"[leg] set zero sent to {self.names[i]} (motor {motor_id})", flush=True)

    def _enable(self):
        now = time.perf_counter()
        with self.lock:
            if self.leg_state != STOPPED:
                self.message = f"enable ignored, leg is {self.leg_state}"
                return
            n = len(self.names)
            silent = [self.names[i] for i in range(n) if not now - self.rx_time[i] < FRESH_S]
            if silent:
                self.message = "enable refused, no reply from " + ", ".join(silent)
                return
            faulted = [i for i in range(n) if self.fault_bits[i]]
            if faulted:
                for i in faulted:
                    self.bus.send_frame(*rp.pack_stop(self.motor_ids[i], clear_fault=True, host_id=self.host_id))

                self.message = "fault clear sent to " + ", ".join(self.names[i] for i in faulted) + ", press Enable again"
                return
            # outside = [f"{self.names[i]} ({self.q[i]:+.2f} rad)" for i in range(n) if abs(self.q[i]) > ANGLE_LIMIT]
            # if outside:
            #     self.message = "enable refused, outside the +-3.14 rad limit: " + ", ".join(outside)
                return
            # hold where the leg already is: first target = measured angle
            self.target = self.q.copy()
            target = self.target.copy()
            speed, accel = self.speed, self.accel
            self.run_mode[:] = -1

        # PP sequence from the RS02 manual (4.3.5): run_mode = 1 -> enable -> vel_max -> acc_set -> loc_ref
        for i, motor_id in enumerate(self.motor_ids):
            self.bus.send_frame(*rp.pack_param_write_uint8(motor_id, rp.RUN_MODE_INDEX, rp.RUN_MODE_PP, self.host_id))
            self.bus.send_frame(*rp.pack_enable(motor_id, self.host_id))
            self.bus.send_frame(*rp.pack_param_write_float(motor_id, rp.VEL_MAX_INDEX, speed, self.host_id))
            self.bus.send_frame(*rp.pack_param_write_float(motor_id, rp.ACC_SET_INDEX, accel, self.host_id))
            self.bus.send_frame(*rp.pack_param_write_float(motor_id, rp.LOC_REF_INDEX, target[i], self.host_id))
            self.bus.send_frame(*rp.pack_param_read(motor_id, rp.RUN_MODE_INDEX, self.host_id))   # check it took PP
        if DEBUG:
            print(f"[leg] enable sent: run_mode {rp.RUN_MODE_PP}, vel_max {speed}, acc_set {accel}, loc_ref {np.round(target, 3)}", flush=True)
        with self.lock:
            self.leg_state = ENABLED
            self.message = f"enabled in PP mode, holding current pose (speed {speed} rad/s, accel {accel} rad/s^2)"
