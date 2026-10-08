import math as m
import numpy as np
from LOGIC.GaitPlanner import GaitPlanner, LEG_NAMES, FORWARD
from LOGIC.FOSMCLogic import FOSMC

FOOT_R = 0.014    # foot sphere radius: foot center height when touching the floor

# Task vector layout (18 rows), shared with the FOSMC and the WBC:
# 0:3 body x y z | 3:6 body rotation | 6:18 feet FL FR BL BR (x y z each)
N_TASK = 18

# Acceleration-level PD gains (Kp, Kd), Kd = 2 * sqrt(Kp) = no overshoot
DEFAULT_PD = {
    "base_z":   (100.0, 20.0),
    "base_xy":  (50.0, 14.0),
    "base_ang": (200.0, 28.0),
    "swing":    (900.0, 60.0),
    "posture":  (50.0, 14.0),
}


def yaw_rot(psi):
    c, s = m.cos(psi), m.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def so3_log(R):
    # NOTE: rotation matrix -> rotation vector (axis * angle, radians)
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = m.acos(cos_th)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    return (th / (2.0 * m.sin(th))) * w


def _pd_rows(g):
    # Group PD gains -> one (Kp, Kd) per task row
    kp, kd = np.zeros(N_TASK), np.zeros(N_TASK)
    kp[0:2], kd[0:2] = g["base_xy"]
    kp[2], kd[2] = g["base_z"]
    kp[3:6], kd[3:6] = g["base_ang"]
    kp[6:], kd[6:] = g["swing"]
    return kp, kd


class TaskController:
    """Turns gait targets + robot state into task accelerations for the WBC.
    Simulator-agnostic: each simulator only supplies the state."""

    def __init__(
        self,
        logic,
        nominal_stance,
        z_nom,
        foot_nom_xy,
        leash=0.1,
        k_foot=0.2,
        gains=None,
        use_fosmc=False,
        fosmc_dt=0.002,
        aw_abs=1.0,
        aw_rel=0.2,
        walk_drop=0.0,
        yaw_leash=0.2,
    ):
        self.logic = logic
        self.q_nom = np.asarray(nominal_stance, dtype=float)
        self.z_nom = z_nom
        self.foot_nom_xy = np.asarray(foot_nom_xy, dtype=float)   # 4 x 2, body frame, FL FR BL BR
        logic.nominal_feet = {leg: self.foot_nom_xy[i].copy() for i, leg in enumerate(LEG_NAMES)}
        self.leash = leash          # max distance the body target may run ahead (m)
        self.k_foot = k_foot        # foot placement correction, ~ sqrt(height / g)
        self.g = dict(DEFAULT_PD)
        if gains:
            self.g.update(gains)
        self.kp, self.kd = _pd_rows(self.g)

        self.fosmc = FOSMC(N_TASK, fosmc_dt) if use_fosmc else None
        self.aw_abs = aw_abs        # anti-windup: freeze a row when achieved and commanded
        self.aw_rel = aw_rel        # differ by more than aw_abs + aw_rel * |command|
        self.last_cmd = None

        self.base_xy_d = None
        self.yaw_d = 0.0
        self.was_turning = False
        self.yaw_leash = yaw_leash
        self.v_cmd = np.zeros(2)
        self.debug = {}

        self.walk_drop = walk_drop
        self.q_walk = self._lowered_pose(walk_drop)
        self.blend = 0.0

    def command_velocity(self):
        # NOTE: same formula the MPC used: stride covered during stance, body -x is forward
        # NOTE: stride covered during stance, along the robot's front (FORWARD in GaitPlanner)
        lg = self.logic
        if lg.walking and isinstance(lg.planner, GaitPlanner):
            gp = lg.planner.p
            v_fwd = FORWARD * (gp.step_len * lg.planner.stride_scale * gp.freq) / gp.duty
            return v_fwd * np.array([m.cos(lg.target_yaw), m.sin(lg.target_yaw)])
        return np.zeros(2)

    def _lowered_pose(self, drop):
        # NOTE: joint angles that keep each foot at the same x, y as NOMINAL_STANCE
        # but `drop` m closer to the body (same knee direction as the nominal pose)
        if drop == 0.0:
            return self.q_nom.copy()
        kin = self.logic.kinematics
        q = []
        for i, leg in enumerate(LEG_NAMES):
            H = kin.fk(leg, *np.degrees(self.q_nom[3 * i: 3 * i + 3]))   # fk takes degrees
            x, y, z = H[0:3, 3]
            knee_dir = -int(np.sign(self.q_nom[3 * i + 2]))   # IK seed on the same knee side as the nominal pose
            q.extend(kin.ik(leg, x, y, z + drop, knee_dir=knee_dir))
        return np.array(q)

    def compute(self, t, dt, p, R, v_base, q, v_joints, pf, vf, acc_achieved=None):
        """Returns the keyword arguments for wbc.solve().
        acc_achieved: last WBC result [acc_base (6), acc_foot (12)], used for anti-windup."""
        lg = self.logic
        tg = lg.wbc_targets(t)
        self.v_cmd = self.command_velocity()

        e = np.zeros(N_TASK)
        e_dot = np.zeros(N_TASK)
        acc_ff = np.zeros(N_TASK)
        active = np.ones(N_TASK, dtype=bool)

        # ---- Body x, y: leashed moving target ----
        if self.base_xy_d is None:
            self.base_xy_d = p[0:2].copy()
        self.base_xy_d += self.v_cmd * dt
        gap = self.base_xy_d - p[0:2]
        dist = np.linalg.norm(gap)
        if dist > self.leash:
            self.base_xy_d = p[0:2] + gap / dist * self.leash
        e[0:2] = p[0:2] - self.base_xy_d
        e_dot[0:2] = v_base[0:2] - self.v_cmd

        # ---- Body height ----
        goal = 1.0 if (lg.walking or lg.turning) else 0.0
        self.blend += float(np.clip(goal - self.blend, -dt, dt))
        e[2] = p[2] - (self.z_nom - self.blend * self.walk_drop)
        e_dot[2] = v_base[2]

        # ---- Body rotation: level, facing target_yaw ----
        # WALK / stand: GaitLogic owns the heading. TURN: integrate the yaw rate here
        # every WBC step, leashed to the real yaw (same idea as the body x, y leash)
        yaw_rate = lg.yaw_rate if lg.turning else 0.0
        if lg.turning and self.was_turning:
            self.yaw_d += yaw_rate * dt
            gap = (self.yaw_d - m.atan2(R[1, 0], R[0, 0]) + m.pi) % (2.0 * m.pi) - m.pi
            if abs(gap) > self.yaw_leash:
                self.yaw_d += m.copysign(self.yaw_leash, gap) - gap
            lg.target_yaw = self.yaw_d      # leaving TURN keeps the reached heading
        else:
            self.yaw_d = lg.target_yaw
        self.was_turning = lg.turning
        R_d = yaw_rot(self.yaw_d)
        e[3:6] = so3_log(R @ R_d.T)     # world frame: how far R is rotated past R_d
        w_d = np.array([0.0, 0.0, yaw_rate])
        e_dot[3:6] = v_base[3:6] - w_d

        # ---- Feet: planner path placed in the world (stance rows are inactive) ----
        R_yaw = R_d[0:2, 0:2]
        v_err = v_base[0:2] - self.v_cmd
        pf_d = np.zeros(12)
        for i in range(4):
            r = slice(3 * i, 3 * i + 3)
            tr = slice(6 + 3 * i, 9 + 3 * i)
            off, vel, acc = tg["off"][i], tg["vel"][i], tg["acc"][i]

            rel = R_yaw @ (self.foot_nom_xy[i] + off[0:2])   # foot from body center, world frame
            xy_d = p[0:2] + rel
            # foot placement correction (Raibert): 0 at lift-off, full at touchdown
            xy_d = xy_d + self.k_foot * tg["s"][i] * v_err

            # rel turns with the yaw target, so add its spin to the feed-forward terms
            wz = w_d[2]
            vel_w = R_yaw @ vel[0:2]
            spin_v = wz * np.array([-rel[1], rel[0]])                              # w x rel
            spin_a = 2.0 * wz * np.array([-vel_w[1], vel_w[0]]) - wz ** 2 * rel   # Coriolis + centripetal

            pos_d = np.array([xy_d[0], xy_d[1], FOOT_R + off[2]])
            vel_d = np.concatenate([self.v_cmd + vel_w + spin_v, [vel[2]]])
            acc_d = np.concatenate([R_yaw @ acc[0:2] + spin_a, [acc[2]]])
            pf_d[r] = pos_d

            if tg["stance"][i] > 0.5:
                active[tr] = False
            else:
                e[tr] = pf[r] - pos_d
                e_dot[tr] = vf[r] - vel_d
                acc_ff[tr] = acc_d

        # ---- Controller: task errors -> commanded task accelerations ----
        # miss: how far the WBC's achieved accelerations were from last step's command (any mode)
        miss = np.zeros(N_TASK)
        frozen = np.zeros(N_TASK, dtype=bool)
        if acc_achieved is not None and self.last_cmd is not None:
            miss = np.abs(acc_achieved - self.last_cmd)
            frozen = active & (miss > self.aw_abs + self.aw_rel * np.abs(self.last_cmd))

        s = np.zeros(N_TASK)
        if self.fosmc is None:
            acc_cmd = acc_ff - self.kp * e - self.kd * e_dot
        else:
            acc_cmd = self.fosmc.compute(e, e_dot, acc_ff, active=active, freeze=frozen)
            s = self.fosmc.s.copy()
        s_max = float(np.max(np.abs(s[active])))
        acc_cmd[~active] = 0.0
        self.last_cmd = acc_cmd.copy()

        # ---- Posture: weak pull toward the nominal stance ----
        kp, kd = self.g["posture"]
        q_post = (1.0 - self.blend) * self.q_nom + self.blend * self.q_walk
        a_joint_ref = kp * (q_post - q) - kd * v_joints

        fc_ref = np.zeros(12)
        fc_ref[2::3] = tg["fz"]

        self.debug = {"pf_d": pf_d, "stance": tg["stance"], "base_xy_d": self.base_xy_d.copy(),
                      "s_max": s_max, "frozen": int(frozen.sum()),
                      "s": s, "miss": miss, "active": active.copy()}
        return dict(stance=tg["stance"], acc_base=acc_cmd[0:6], acc_foot=acc_cmd[6:],
                    a_joint_ref=a_joint_ref, fc_ref=fc_ref)
