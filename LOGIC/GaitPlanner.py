import math as m
from dataclasses import dataclass
import numpy as np

LEG_NAMES = ['FL', 'FR', 'BL', 'BR']
GRAVITY = 9.81
FORWARD = 1

class BasePlanner:
    def phase(self, leg, t): 
        raise NotImplementedError

    def foot_offset(self, leg, t):
        raise NotImplementedError

    def fz_scale(self, leg, t):
        return 1.0

    def vz_active(self, leg, t):
        return 1.0

    def xy_active(self, leg, t):
        return 0.0

    def foot_vel(self, leg, t, h=1e-4):
        # NOTE: Central difference of foot offset, so the velocity always matches
        # whatever path shape the subclass defines
        p1 = self.foot_offset(leg, t + h)
        p0 = self.foot_offset(leg, t - h)
        return tuple((a - b) / (2.0 * h) for a, b in zip(p1, p0))

    def foot_acc(self, leg, t, h=1e-3):
        p1 = self.foot_offset(leg, t + h)
        p = self.foot_offset(leg, t)
        p0 = self.foot_offset(leg, t - h)
        return tuple(max(-50.0, min(50.0, (a - 2.0 * b + c) / (h * h))) for a, b, c in zip(p1, p, p0))

    def horizon(self, t0, n, dt, mass):
        # NOTE: Mpc Schedule for nodes k = 0..n-1 at t0 + k*dt, 
        # each as a list over LEG_NAMES: stance flags, swing vz, vz mask
        # and Fc guess [fx, fy, fz]

        out = {"stance": [], "swing_vz": [], "vz_active": [], "fc_guess": [], "swing_xy": [], "xy_active": []}
        for k in range(n):
            t_k = t0 + k * dt
            k_stance = [self.phase(leg, t_k)[0] for leg in LEG_NAMES]
            stance_count = sum(k_stance)
            base = mass * GRAVITY / stance_count if stance_count > 0 else 0.0

            k_vz, k_mask, k_fc, k_xy, k_xya = [], [], [], [], []
            for leg, is_stance in zip(LEG_NAMES, k_stance):
                if is_stance:
                    k_vz.append(0.0)
                    k_fc.append([0.0, 0.0, base * self.fz_scale(leg, t_k)])
                    k_xy.append([0.0, 0.0])
                else:
                    k_vz.append(self.foot_vel(leg, t_k)[2])
                    k_fc.append([0.0, 0.0, 0.0])
                    k_xy.append(list(self.foot_offset(leg, t_k)[0:2]))
                k_mask.append(self.vz_active(leg, t_k))
                k_xya.append(self.xy_active(leg, t_k))

            out["stance"].append(k_stance)
            out["swing_vz"].append(k_vz)
            out["vz_active"].append(k_mask)
            out["fc_guess"].append(k_fc)
            out["swing_xy"].append(k_xy)
            out["xy_active"].append(k_xya)

        return out

    def instant(self, t, mass):
        # NOTE: Single-instant schedule for the WBC, per leg in LEG_NAMES order:
        # stance flag, phase progress s, foot offset / velocity / acceleration, Fz guess
        stance, s, off, vel, acc = [], [], [], [], []
        for leg in LEG_NAMES:
            is_stance, s_leg = self.phase(leg, t)
            stance.append(1.0 if is_stance else 0.0)
            s.append(s_leg)
            off.append(self.foot_offset(leg, t))
            vel.append(self.foot_vel(leg, t))
            acc.append(self.foot_acc(leg, t))

        n_st = sum(stance)
        fz = [mass * GRAVITY / n_st * self.fz_scale(leg, t) * st if n_st > 0 else 0.0
              for leg, st in zip(LEG_NAMES, stance)]

        return {
            "stance": np.array(stance),
            "s": np.array(s),
            "off": np.array(off, dtype=float),
            "vel": np.array(vel, dtype=float),
            "acc": np.array(acc, dtype=float),
            "fz": np.array(fz),
        }

class StandPlanner(BasePlanner):
    def phase(self, leg, t):
        return True, 0.0

    def foot_offset(self, leg, t):
        return 0.0, 0.0, 0.0

@dataclass
class GaitParams:
    freq: float             # gait cycles per seconds
    duty: float             # stance fraction of the cycles [0..1]
    offsets: dict           # each leg action offsets, fraction of each cycle [0..1]
    path: str               # foot trajectory shape : "WALK" | "RUN" | "TURN"
    step_len: float         # stride / step length 
    step_h: float           # swing height
    x_off: float            # stance posture x (front..back) offsets
    z_off: float            # stance posture heights offsets 
    sc_yaw: float = 0.0     # turn control, yaw rate is used when NONE is the state
    fz_gain: float = 1.0    # stance force multipler (for RUN to push even harder)

GAIT_TABLE = {
    "WALK": GaitParams(
        freq=2.6,
        duty=0.5,
        offsets= {
            'FL': 0.0,
            'BR': 0.0,
            'FR': 0.5,
            'BL': 0.5,
        },
        path="WALK",
        step_len=0.073,
        step_h=0.08,
        x_off=0.0,
        z_off=0.365,
    ),
    "CRAWL": GaitParams(
        freq=2.0,
        duty=0.5,
        offsets= {
            'FL': 0.0,
            'BR': 0.0,
            'FR': 0.5,
            'BL': 0.5,
        },
        path="WALK",
        step_len=0.117,
        step_h=0.073,
        x_off=0.051,
        z_off=0.263,
    ),
    "RUN": GaitParams(
        freq=2.75,
        duty=0.5,
        offsets= {
            'FL': 0.0,
            'BR': 0.5,
            'FR': 0.0,
            'BL': 0.5,
        },
        path="RUN",
        step_len=0.438,
        step_h=0.292,
        x_off=0.11,
        z_off=0.365,
        fz_gain=1.75,
    ),
    "TURN": GaitParams(
        freq=1.0,
        duty=0.5,
        offsets= {
            'FL': 0.0,
            'BR': 0.5,
            'FR': 0.5,
            'BL': 0.0,
        },
        path="TURN",
        step_len=0.0,
        step_h=0.036,
        x_off=0.0,
        z_off=0.365,
        sc_yaw= 0.5,
    )
}

class GaitPlanner(BasePlanner):
    # NOTE: Periodic Gaits (WALK | CRAWL | RUN | TURN) as one planner with different Params

    def __init__(self, params: GaitParams, nominal_feet: dict):
        self.p = params
        self.nominal_feet = nominal_feet

        self.stride_scale = 1.0
        self.yaw_rate = 0.0

    def phase(self, leg, t):
        cycle = (t * self.p.freq + self.p.offsets[leg]) % 1.0
        if cycle < self.p.duty:
            return True, cycle / self.p.duty
        return False, (cycle - self.p.duty) / (1.0 - self.p.duty)

    def fz_scale(self, leg, t):
        return self.p.fz_gain

    def xy_active(self, leg, t):
        return 1.0 if self.p.path == "WALK" else 0.0

    def foot_offset(self, leg, t):
        is_stance, s = self.phase(leg, t)
        if self.p.path == "TURN":
            return self._turn_path(leg, is_stance, s)
        if self.p.path == "RUN":
            x, y, z = self._run_path(leg, is_stance, s)
        else:
            x, y, z = self._walk_path(is_stance, s)
        # NOTE: the paths are written for a front at -x (stance foot sweeps toward +x);
        # mirror x so the stance foot always sweeps away from the front
        return -FORWARD * x, y, z

    def _lift(self, s):
        # NOTE: Swing polynomial trajectory
        return 64.0 * self.p.step_h * (s**3) * ((1.0 - s)**3)

    def _walk_path(self, is_stance, s):
        l = self.p.step_len * self.stride_scale
        if is_stance:
            return -(l / 2.0) + s * l, 0.0, 0.0

        v = l * ((1.0 - self.p.duty) / self.p.duty)
        c = l + v
        d = (l / 2.0) + v * s - 10.0 * c * s**3 + 15.0 * c * s**4 - 6.0 * c * s**5
        return d, 0.0, self._lift(s)

    def _run_path(self, leg, is_stance, s):
        l = self.p.step_len * self.stride_scale

        if leg in ('BL', 'BR'):
            x_start, x_end = 0.8 * l, 0.2 * l
        else:
            x_start, x_end = 0.2 * l, 0.8 * l
        if is_stance:
            return x_start + s * (x_end - x_start), 0.0, 0.0

        target_x = -x_end + s * (x_end - x_start)
        d = (1.0 - s / 0.15) * x_end + (s / 0.15) * target_x if s < 0.15 else target_x
        return d, 0.0, self._lift(s)

    def _turn_path(self, leg, is_stance, s):
        ix, iy = self.nominal_feet[leg]
        yaw_rate = self.yaw_rate if abs(self.yaw_rate) >= 0.1 else self.p.sc_yaw
        sweep = yaw_rate * (self.p.duty / self.p.freq)
        r = m.hypot(ix, iy)
        base_angle = m.atan2(iy, ix)
        if is_stance:
            angle_offset = (0.5 - s) * sweep
            dz = 0.0
        else:
            c_s = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            angle_offset = (-0.5 + c_s) * sweep
            dz = self._lift(s)
        a = base_angle + angle_offset
        return r * m.cos(a) - ix, r * m.sin(a) - iy, dz

JUMP_SEQ = (
    "PREPARE",
    "THRUST",
    "FLIGHT",
    "DESCENT",
    "LANDING"
)

@dataclass
class JumpParams:
    x_off: float = 0.25
    z_off: float = 2.7
    y_crouch: float = 1.0
    y_thrust: float = 3.8
    y_flight: float = 1.8
    x_flight: float = -1.5
    x_catch: float = -1.5
    x_stabilize: float = 1.5
    back_thrust: float = 1.5
    prepare_time: float = 0.8
    back_thrust_time: float = 0.15
    flight_time: float = 0.05
    catch_time: float = 0.1
    landing_time: float = 0.5

class JumpPlanner(BasePlanner):
    """
    One-shot jump: PREPARE -> THRUST -> FLIGHT -> DESCENT -> LANDING.

    Pure-time like GaitPlanner, but two transitions are events GaitLogic
    detects (crouch height reached, early touchdown). GaitLogic reports them
    with start_phase(name, t), which re-anchors that phase and every phase
    after it. Until THRUST is started, the timeline holds in PREPARE.

    foot_offset returns (x, y, 0.0) in the jump's own IK mapping, the same
    as jump_process: tx = ix + x, ty = y, tz = iz (identical for all legs).
    """

    # (stance, fz scale) per phase, matching jump_process foot_forces
    CONTACT = {
        "PREPARE": (True, 1.0),
        "THRUST": (True, 2.5),
        "FLIGHT": (False, 0.0),
        "DESCENT": (False, 0.0),
        "LANDING": (True, 1.5),
    }

    def __init__(self, params: JumpParams):
        self.p = params
        self.reset()

    # ---- timeline ----

    def durations(self):
        p = self.p
        return {
            "PREPARE": p.prepare_time,
            "THRUST": p.back_thrust_time,
            "FLIGHT": p.flight_time,
            "DESCENT": p.catch_time,
            "LANDING": p.landing_time,
        }

    def reset(self):
        # Default timeline from t = 0, PREPARE held until released
        self.starts = {}
        t = 0.0
        for name in JUMP_SEQ:
            self.starts[name] = t
            t += self.durations()[name]
        self.released = False

    def start_phase(self, name, t):
        # Re-anchor `name` at t and shift every later phase with it
        dur = self.durations()
        i = JUMP_SEQ.index(name)
        for later in JUMP_SEQ[i:]:
            self.starts[later] = t
            t += dur[later]
        if name != "PREPARE":
            self.released = True

    def jump_phase(self, t):
        if not self.released:
            return "PREPARE"
        current = "PREPARE"
        for name in JUMP_SEQ:
            if t >= self.starts[name]:
                current = name
        return current

    def phase_end(self, name):
        return self.starts[name] + self.durations()[name]

    def finished(self, t):
        return self.released and t >= self.phase_end("LANDING")

    # ---- posture (the old jump_process curves) ----

    def posture(self, name, t):
        # Evaluate one phase's curve at t without re-deciding the phase, so
        # GaitLogic can lock the phase for its central difference. The
        # fraction is deliberately not clipped (extrapolates, as before).
        p = self.p
        y_idle, x_idle = p.z_off, p.x_off
        x_stab = p.x_off + p.x_stabilize
        x_thrust = p.x_off + p.back_thrust
        x_flight = p.x_off + p.x_flight
        x_catch = p.x_off + p.x_catch

        f = (t - self.starts[name]) / self.durations()[name]
        smooth = 3.0 * f**2 - 2.0 * f**3

        if name == "PREPARE":
            return x_idle + smooth * (x_stab - x_idle), y_idle + smooth * (p.y_crouch - y_idle)
        if name == "THRUST":
            s = f**2
            return x_stab + s * (x_thrust - x_stab), p.y_crouch + s * (p.y_thrust - p.y_crouch)
        if name == "FLIGHT":
            return x_thrust + f * (x_flight - x_thrust), p.y_thrust + f * (p.y_flight - p.y_thrust)
        if name == "DESCENT":
            return x_flight + smooth * (x_catch - x_flight), p.y_flight
        # LANDING
        return x_catch + smooth * (x_idle - x_catch), p.y_flight + smooth * (y_idle - p.y_flight)

    # ---- BasePlanner interface ----

    def phase(self, leg, t):
        name = self.jump_phase(t)
        f = (t - self.starts[name]) / self.durations()[name]
        return self.CONTACT[name][0], min(max(f, 0.0), 1.0)

    def foot_offset(self, leg, t):
        x, y = self.posture(self.jump_phase(t), t)
        return x, y, 0.0

    def fz_scale(self, leg, t):
        return self.CONTACT[self.jump_phase(t)][1]

    def vz_active(self, leg, t):
        # In flight the feet ride with the ballistic base: leave vz free
        return 1.0 if self.CONTACT[self.jump_phase(t)][0] else 0.0
