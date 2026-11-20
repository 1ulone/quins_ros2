import os
import sys
import time
import threading
from dataclasses import replace
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(current_dir)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from HARDWARE.LegDriver import ENABLED
from LOGIC.GaitPlanner import GaitPlanner, GaitParams, GAIT_TABLE
from LOGIC.KinematicsLogic import KinematicsLogic

LEG_NAMES = ("FL", "FR", "BL", "BR")         # names GaitPlanner and KinematicsLogic use
JOINT_NAMES = ("HAA", "HFE", "KFE")
GAIT = "WALK_4BEAT"

# 4-beat walk: one leg in the air at a time, order FL -> BR -> FR -> BL, each a quarter cycle after the last.
# Step length and height as WALK; freq 1.3 keeps each swing as long as a WALK swing (0.25 / 1.3 = 0.5 / 2.6 = 0.19 s)
HARDWARE_GAITS = {
    "WALK_4BEAT": GaitParams(
        freq=1.3,
        duty=0.75,
        offsets={"FL": 0.0, "BR": 0.25, "FR": 0.5, "BL": 0.75},
        path="WALK",
        step_len=0.073,
        step_h=0.04,
        x_off=0.0,
        z_off=0.365,
    ),
    # step in place (no stride), 4-beat order; while a leg swings the body tilts away from it (see GaitPlayer.tilt_drop)
    "TILT_4BEAT": GaitParams(
        freq=1.0,
        duty=0.75,
        offsets={"FL": 0.0, "BR": 0.25, "FR": 0.5, "BL": 0.75},
        path="WALK",
        step_len=0.1,
        step_h=0.1,
        x_off=0.0,
        z_off=0.365,
    ),
    "WALK": GaitParams(
        freq=2.6,
        duty=0.75,
        offsets= {'FL': 0.0, 'BR': 0.0, 'FR': 0.5, 'BL': 0.5,},
        path="WALK",
        step_len=0.073,
        step_h=0.04,
        x_off=0.0,
        z_off=0.365,
    ),
}

# serial port and motor IDs (HAA, HFE, KFE) per leg
LEG_SETUP = {
    "FL": ("/dev/ttyUSB0", [11, 12, 13]),
    "FR": ("/dev/ttyUSB1", [21, 22, 23]),
    "BL": ("/dev/ttyUSB2", [31, 32, 33]),
    "BR": ("/dev/ttyUSB3", [41, 42, 43]),
}

# stand pose per leg, URDF rad (HAA, HFE, KFE): Stand goes here and the gait moves around it (measured on the robot)
URDF_REST_RAD = {
    "FL": np.array([0.0, 0.1708, -0.4916]),
    "FR": np.array([0.0, 0.1708, -0.4916]),
    "BL": np.array([0.0, -0.1708, 0.4916]),
    "BR": np.array([0.0, -0.1708, 0.4916]),
}

WALK_DROP_M = 0.01

# rest pose per leg, URDF rad: about 0.05 rad short of the limp readings (HFE -1.48 .. -1.50, KFE 2.70 .. 2.88),
# so no motor pushes into its stop; the legs settle the rest of the way when the motors turn off

# stand pose in motor rad, counted from the start-up zero (the robot lying at rest): taught on the robot,
# read from the GUI. Rest itself is motor 0, so it needs no numbers
STAND_MOTOR_RAD = {
    "FL": np.array([0.0, 0.350, -1.15]),
    "FR": np.array([0.0, -0.350, 1.15]),
    "BL": np.array([0.0, -0.350, 1.15]),
    "BR": np.array([0.0, 0.350, -1.15]),
}

# motor = sign * q_urdf + offset, per joint
MOTOR_SIGN = {
    "FL": np.array([-1.0, 1.0, 1.0]),
    "FR": np.array([1.0, -1.0, -1.0]),
    "BL": np.array([-1.0, 1.0, 1.0]),        # HFE, KFE confirmed by the rest / stand readings; HAA not checked yet
    "BR": np.array([1.0, -1.0, -1.0]),       # HFE, KFE confirmed by the rest / stand readings; HAA not checked yet
}
MOTOR_OFFSET = {leg: -MOTOR_SIGN[leg] * URDF_REST_RAD[leg] for leg in LEG_NAMES}     # fixed by the rest pose: motor 0 = URDF_REST_RAD
URDF_INIT_RAD = {leg: (STAND_MOTOR_RAD[leg] - MOTOR_OFFSET[leg]) / MOTOR_SIGN[leg] for leg in LEG_NAMES}   # stand pose in URDF rad, follows STAND_MOTOR_RAD

# URDF joint limits (rad) for the gait, same as WholeBodyController. The back stand knee (0.825) is under Q_LO[2]
Q_LO = np.array([-np.pi + 0.05, -np.pi + 0.05, -2.967])
Q_HI = np.array([np.pi - 0.05, np.pi - 0.05, 2.967])

TIME_SCALE = 0.25                # default in the GUI; 1.0 = real gait speed
TIME_SCALE_MAX = 0.5
RATE_HZ = 100.0                  # one frame for all legs every 10 ms
STRIDE_RAMP_S = 1.0              # stride grows 0 -> full over the first second of gait time
CYCLE_SAMPLES = 100              # samples in the one-cycle preview
START_SPEED = 0.5                # rad/s, speed of the farthest joint in every ramp (Stand, Rest, return after STOP GAIT)
GAIT_SPEED_MAX = 2.0             # rad/s, no motor target changes faster than this during the gait
START_TOL_RAD = 0.05             # every joint must be this close to the pose before the next step
START_SETTLE_S = 3.0             # time allowed for measured to catch up after a ramp
STANCE_CHECK_RAD = 0.01          # FK -> IK round trip of the stand pose must come back this close
IK_TOL_M = 0.005                 # IK result further than this from the foot target is rejected
TURN_WINDOW_CENTRE_RAD = 0.5 * (Q_LO + Q_HI)   # URDF rad: a joint's true angle is taken as the reading within half a turn of this

# tilt gaits: while one leg swings, its two neighbour legs lower by TILT_ADJACENT_DROP_M and the diagonal leg by
# TILT_ADJACENT_DROP_M + TILT_DIAGONAL_EXTRA_M (m, foot closer to the body), full at mid-swing, 0 at lift-off and touchdown
TILT_GAITS = ("TILT_4BEAT",)
DIAGONAL_LEG = {"FL": "BR", "FR": "BL", "BL": "FR", "BR": "FL"}
TILT_ADJACENT_DROP_M = 0.0000       # default in the GUI
TILT_DIAGONAL_EXTRA_M = 0.0000     # default in the GUI
TILT_DROP_MAX_M = 1.0


def urdf_to_motor(leg, q_urdf):
    return MOTOR_SIGN[leg] * q_urdf + MOTOR_OFFSET[leg]

def step_toward(current, desired, max_step):
    # move each joint from current toward desired by at most max_step (rad)
    return current + np.clip(desired - current, -max_step, max_step)


class GaitPlayer:
    """Stand, Rest and gait for every leg in `drivers` on one clock. Gait: stand xyz + keyframe -> IK -> motor targets.
    Runs its own thread; the GUI only calls start / stand / rest / stop / status.
    stop() during the gait ramps every leg back to the stand pose; during a ramp it holds in place."""

    def __init__(self, drivers=None, legs=LEG_NAMES):
        self.drivers = dict(drivers) if drivers else {}
        self.legs = list(self.drivers) if self.drivers else list(legs)
        self.kin = KinematicsLogic(os.path.join(ROOT, "urdf", "quadruped.urdf"))

        # turn on: stand angles -> foot xyz (FK), then check that IK of that xyz gives the stand pose back
        self.knee_dir, self.foot_home = {}, {}
        for leg in self.legs:
            stance = URDF_INIT_RAD[leg]
        # turn on: stand angles -> foot xyz (FK), check that IK of that xyz gives the stand pose back,
        # then the walk pose: same foot x, y, WALK_DROP_M closer to the body (same as TaskController._lowered_pose)
        self.knee_dir, self.foot_home, self.walk_rad = {}, {}, {}
        for leg in self.legs:
            stance = URDF_INIT_RAD[leg]
            self.knee_dir[leg] = -int(np.sign(stance[2]))                    # IK seed on the stand knee side
            foot_stand = self.kin.fk(leg, *np.degrees(stance))[0:3, 3]       # fk takes degrees
            q_check = np.array(self.kin.ik(leg, *foot_stand, knee_dir=self.knee_dir[leg]))
            if np.max(np.abs(q_check - stance)) > STANCE_CHECK_RAD:
                raise RuntimeError(f"{leg}: IK of the stand xyz gives {np.round(q_check, 3)}, not {stance}")
            self.foot_home[leg] = foot_stand + np.array([0.0, 0.0, WALK_DROP_M])     # the gait keyframes are added to this
            self.walk_rad[leg] = np.array(self.kin.ik(leg, *self.foot_home[leg], knee_dir=self.knee_dir[leg]))
            foot_reached = self.kin.fk(leg, *np.degrees(self.walk_rad[leg]))[0:3, 3]
            if np.linalg.norm(foot_reached - self.foot_home[leg]) > IK_TOL_M:
                raise RuntimeError(f"{leg}: IK did not reach the walk pose")
        gait_params = HARDWARE_GAITS[GAIT] if GAIT in HARDWARE_GAITS else GAIT_TABLE[GAIT]
        self.planner = GaitPlanner(replace(gait_params), {leg: self.foot_home[leg][0:2] for leg in self.legs})
        self.tilt_adjacent_m = TILT_ADJACENT_DROP_M        # set from the GUI with set_tilt
        self.tilt_diagonal_extra_m = TILT_DIAGONAL_EXTRA_M
        self.motor_start = {leg: urdf_to_motor(leg, URDF_INIT_RAD[leg]) for leg in self.legs}
        self.motor_walk = {leg: urdf_to_motor(leg, self.walk_rad[leg]) for leg in self.legs}
        self.motor_rest = {leg: np.zeros(3) for leg in self.legs}                       # rest = the start-up zero

        # one full-stride cycle per leg: what each joint will be asked for (shown in the preview)
        self.cycle_t = np.linspace(0.0, 1.0 / self.planner.p.freq, CYCLE_SAMPLES, endpoint=False)
        self.cycle_q_sim, self.cycle_motor, self.cycle_peak_speed, self.cycle_rejects = {}, {}, {}, {}
        for leg in self.legs:
            q_list, q_prev, rejects = [], self.walk_rad[leg], 0
            for t_k in self.cycle_t:
                q_prev, ok = self.joint_angles(leg, t_k, q_prev, stride_scale=1.0)
                q_list.append(q_prev)
                rejects += int(not ok)
            self.cycle_q_sim[leg] = np.array(q_list)
            self.cycle_motor[leg] = urdf_to_motor(leg, self.cycle_q_sim[leg])
            self.cycle_peak_speed[leg] = np.max(np.abs(np.gradient(self.cycle_motor[leg], self.cycle_t, axis=0)), axis=0)
            self.cycle_rejects[leg] = rejects

        self.lock = threading.Lock()            # guards the fields below (gait thread writes, GUI reads)
        self.state = "IDLE"                     # IDLE, MOVING_TO_START, MOVING_TO_REST, PLAYING, RETURNING, HOLDING
        self.message = "press Enable, then Stand, Start gait or Rest"
        self.t_gait = 0.0
        self.q_sim = {leg: URDF_INIT_RAD[leg].copy() for leg in self.legs}
        self.motor_target = {leg: self.motor_start[leg].copy() for leg in self.legs}
        self.ik_rejects = 0
        self.speed_limited_ticks = 0
        self.stop_flag = threading.Event()
        self.thread = None

    def joint_angles(self, leg, t_gait, q_prev, stride_scale):
        # stand xyz + keyframe -> IK -> URDF angles (rad); returns (angles, ok), keeps q_prev if IK misses
        self.planner.stride_scale = stride_scale
        foot_xyz = self.foot_home[leg] + np.array(self.planner.foot_offset(leg, t_gait))
        foot_xyz[2] += self.tilt_drop(leg, t_gait)
        q = np.array(self.kin.ik(leg, *foot_xyz, knee_dir=self.knee_dir[leg]))
        foot_reached = self.kin.fk(leg, *np.degrees(q))[0:3, 3]
        if np.linalg.norm(foot_reached - foot_xyz) > IK_TOL_M:
            return np.array(q_prev, dtype=float), False
        return np.clip(q, Q_LO, Q_HI), True

    def tilt_drop(self, leg, t_gait):
        # tilt gaits only: m this leg's foot moves closer to the body while another leg swings (0 when no leg swings)
        if GAIT not in TILT_GAITS:
            return 0.0
        for swing_leg in LEG_NAMES:
            is_stance, s = self.planner.phase(swing_leg, t_gait)
            if is_stance or swing_leg == leg:
                continue
            swing_shape = 64.0 * s**3 * (1.0 - s)**3                # 0 at lift-off and touchdown, 1 at mid-swing
            drop_m = self.tilt_adjacent_m
            if DIAGONAL_LEG[swing_leg] == leg:
                drop_m += self.tilt_diagonal_extra_m
            return drop_m * swing_shape
        return 0.0

    # ---------------- GUI side ----------------

    def start(self, time_scale, mode="GAIT"):
        # mode "GAIT": ramp to the stand pose, then play. "STAND": ramp to the stand pose and hold.
        # "REST": ramp to the rest pose, then motors off
        if self.thread is not None and self.thread.is_alive():
            self._set(self.state, "already running, press STOP GAIT first")
            return
        if not (0.0 < time_scale <= TIME_SCALE_MAX):
            self._set("IDLE", f"refused: time scale must be above 0 and at most {TIME_SCALE_MAX}")
            return
        not_enabled = [leg for leg in self.legs if self.drivers[leg].status()["leg_state"] != ENABLED]
        if not self.drivers or not_enabled:
            self._set("IDLE", "refused, not enabled: " + ", ".join(not_enabled or ["no legs"]))
            return
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, args=(time_scale, mode), daemon=True)
        self.thread.start()

    def stand(self):
        self.start(TIME_SCALE, mode="STAND")

    def rest(self):
        self.start(TIME_SCALE, mode="REST")

    def set_tilt(self, adjacent_m, diagonal_extra_m):
        # tilt drops in m, used from the next gait tick (also mid-gait)
        if not (0.0 <= adjacent_m <= TILT_DROP_MAX_M and 0.0 <= diagonal_extra_m <= TILT_DROP_MAX_M):
            self._set(self.state, f"refused: tilt drops must be between 0 and {TILT_DROP_MAX_M * 1000.0:.0f} mm")
            return
        self.tilt_adjacent_m = float(adjacent_m)
        self.tilt_diagonal_extra_m = float(diagonal_extra_m)
        self._set(self.state, f"tilt: neighbours {self.tilt_adjacent_m * 1000.0:.0f} mm, diagonal {(self.tilt_adjacent_m + self.tilt_diagonal_extra_m) * 1000.0:.0f} mm")

    def stop(self):
        # during the gait: every leg ramps back to the stand pose, then holds. During a ramp: holds in place
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=0.2)

    def status(self):
        with self.lock:
            return {
                "state": self.state,
                "message": self.message,
                "t_gait": self.t_gait,
                "q_sim": {leg: q.copy() for leg, q in self.q_sim.items()},
                "motor_target": {leg: m.copy() for leg, m in self.motor_target.items()},
                "ik_rejects": self.ik_rejects,
                "speed_limited_ticks": self.speed_limited_ticks,
            }

    # ---------------- gait thread ----------------

    def _set(self, state, message):
        with self.lock:
            self.state = state
            self.message = message

    def _send(self, leg, motor_target):
        for j in range(3):
            self.drivers[leg].set_target_rad(j, motor_target[j])
        with self.lock:
            self.motor_target[leg] = motor_target.copy()

    def _legs_lost(self):
        # one leg out of ENABLED (E-STOP, fault) = every leg e-stopped
        lost = [leg for leg in self.legs if self.drivers[leg].status()["leg_state"] != ENABLED]
        if lost:
            for leg in self.legs:
                self.drivers[leg].estop()
            self._set("IDLE", "not enabled: " + ", ".join(lost) + ", all legs e-stopped")
            return True
        return False

    def _stopped(self):
        if self._legs_lost():
            return True
        if self.stop_flag.is_set():
            self._set("HOLDING", "STOP GAIT: all legs holding, motors on")
            return True
        return False

    def _run(self, time_scale, mode):
        period = 1.0 / RATE_HZ
        cmd = {leg: self.drivers[leg].status()["target"].copy() for leg in self.legs}   # motor rad held now (= measured at Enable)

        # power-on readings are in 0..2pi, so a joint can read a whole turn off: pick the whole-turn shift that
        # puts each joint's current URDF angle inside the one-turn window around the middle of its limits
        turn_shift = {}
        for leg in self.legs:
            q_urdf_now = (cmd[leg] - MOTOR_OFFSET[leg]) / MOTOR_SIGN[leg]
            turns = np.round((q_urdf_now - TURN_WINDOW_CENTRE_RAD) / (2.0 * np.pi))
            turn_shift[leg] = MOTOR_SIGN[leg] * 2.0 * np.pi * turns
        motor_stance = {leg: self.motor_start[leg] + turn_shift[leg] for leg in self.legs}
        motor_walk = {leg: self.motor_walk[leg] + turn_shift[leg] for leg in self.legs}
        motor_rest = {leg: self.motor_rest[leg] + turn_shift[leg] for leg in self.legs}
        next_t = time.perf_counter()

        def wait_next():
            nonlocal next_t
            next_t += period
            wait = next_t - time.perf_counter()
            if wait > 0.0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()

        def ramp_to(motor_goal, state, label, stop_check):
            # every joint of every leg reaches motor_goal in the same number of ticks, the farthest joint at START_SPEED;
            # returns False if stop_check() ended it early
            ramp_from = {leg: cmd[leg].copy() for leg in self.legs}
            farthest_rad = max(float(np.max(np.abs(motor_goal[leg] - ramp_from[leg]))) for leg in self.legs)
            ticks = max(1, int(np.ceil(farthest_rad / (START_SPEED * period))))
            self._set(state, f"{label} at {START_SPEED} rad/s ({ticks * period:.0f} s)")
            for tick in range(1, ticks + 1):
                if stop_check():
                    return False
                for leg in self.legs:
                    cmd[leg] = ramp_from[leg] + (motor_goal[leg] - ramp_from[leg]) * (tick / ticks)
                    self._send(leg, cmd[leg])
                wait_next()
            for leg in self.legs:
                cmd[leg] = motor_goal[leg].copy()
                self._send(leg, cmd[leg])
            return True

        def legs_not_arrived(motor_goal):
            # wait up to START_SETTLE_S for measured to reach motor_goal; returns the legs still off it, None if stopped
            t_end = time.perf_counter() + START_SETTLE_S
            while True:
                if self._stopped():
                    return None
                far = [leg for leg in self.legs
                       if not np.all(np.abs(self.drivers[leg].status()["q"] - motor_goal[leg]) < START_TOL_RAD)]
                if not far or time.perf_counter() > t_end:
                    return far
                wait_next()

        # REST: ramp to the rest pose, then motors off (brake, then stop)
        if mode == "REST":
            if not ramp_to(motor_rest, "MOVING_TO_REST", "moving to the rest pose", self._stopped):
                return
            if legs_not_arrived(motor_rest) is None:
                return
            for leg in self.legs:
                self.drivers[leg].estop()
            self._set("IDLE", "at rest, motors off")
            return

        # 1) STAND and GAIT: ramp to the stand pose, then wait until all measured angles are there
        if not ramp_to(motor_stance, "MOVING_TO_START", "moving to the stand pose", self._stopped):
            return
        far = legs_not_arrived(motor_stance)
        if far is None:
            return
        if far:
            self._set("HOLDING", "stand pose not reached by " + ", ".join(far) + ", holding")
            return
        if mode == "STAND":
            self._set("HOLDING", "standing, motors on")
            return

        # 1b) GAIT: lower from the stand pose (idle) to the walk pose
        if not ramp_to(motor_walk, "MOVING_TO_START", "lowering to the walk pose", self._stopped):
            return

        # 2) play: same t_gait for every leg, the phase offset decides whose turn it is.
        #    STOP GAIT: the stride shrinks to zero while the legs keep stepping, then the gait ends at the
        #    next moment all four feet are down (once every half cycle)
        self._set("PLAYING", f"playing {GAIT} at time scale {time_scale}, joint speed cap {GAIT_SPEED_MAX} rad/s")
        t_gait = 0.0
        stop_t_gait = None                       # gait time when STOP GAIT was pressed
        stride_at_stop = 0.0                     # stride scale at that moment
        half_cycles_per_s = 2.0 * self.planner.p.freq
        q_sim = {leg: self.walk_rad[leg].copy() for leg in self.legs}
        with self.lock:
            self.ik_rejects = 0
            self.speed_limited_ticks = 0
        while True:
            if self._legs_lost():
                return
            if self.stop_flag.is_set():
                break                                    # STOP GAIT: end the gait now, then ramp straight to the stand pose
            stride_scale = min(t_gait / STRIDE_RAMP_S, 1.0)
            limited, rejects = False, 0
            for leg in self.legs:
                q_sim[leg], ok = self.joint_angles(leg, t_gait, q_sim[leg], stride_scale)
                motor_desired = urdf_to_motor(leg, q_sim[leg]) + turn_shift[leg]
                new_cmd = step_toward(cmd[leg], motor_desired, GAIT_SPEED_MAX * period)
                limited = limited or not np.array_equal(new_cmd, motor_desired)
                rejects += int(not ok)
                cmd[leg] = new_cmd
                self._send(leg, cmd[leg])
            with self.lock:
                self.t_gait = t_gait
                self.q_sim = {leg: q.copy() for leg, q in q_sim.items()}
                self.ik_rejects += rejects
                self.speed_limited_ticks += int(limited)
            t_gait += period * time_scale
            wait_next()

        # 3) settle exactly on the stand pose and hold
        if ramp_to(motor_stance, "RETURNING", "STOP GAIT: settling on the stand pose", self._legs_lost):
            self._set("HOLDING", "standing, motors on")


if __name__ == "__main__":
    # no hardware: stand / rest pose and the min / max motor angle and peak speed per joint over one cycle, every leg
    player = GaitPlayer()
    cycle_s = 1.0 / player.planner.p.freq
    print(f"{GAIT}: one cycle = {cycle_s:.3f} s gait time ({cycle_s / TIME_SCALE:.3f} s at time scale {TIME_SCALE})")
    for leg in player.legs:
        print(f"\n{leg}  stand motor {np.round(player.motor_start[leg], 3)} rad   rest motor {np.round(player.motor_rest[leg], 3)} rad   "
              f"IK rejects {player.cycle_rejects[leg]} of {CYCLE_SAMPLES}")
        for j, name in enumerate(JOINT_NAMES):
            motor = player.cycle_motor[leg][:, j]
            print(f"  {name}: motor {motor.min():+.3f} .. {motor.max():+.3f} rad   "
                  f"peak speed {player.cycle_peak_speed[leg][j]:.2f} rad/s at scale 1.0")
