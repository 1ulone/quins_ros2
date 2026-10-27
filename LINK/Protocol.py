import json
import struct
import numpy as np

CTRL_PORT = 9870

KIND_HELLO = b"H"   # sim -> controller: robot constants (JSON)
KIND_READY = b"R"   # controller -> sim: controller built, real time can start
KIND_STATE = b"S"   # sim -> controller: one robot state
KIND_CMD = b"C"     # controller -> sim: torques answering one state
KIND_END = b"E"     # sim -> controller: run finished
READY_PKT = KIND_READY
END_PKT = KIND_END

STATE_NAMES = (
    "TUNING",
    "IDLE",
    "WALK",
    "WALK_BACK",
    "TURN",
    "TURN_RIGHT",
    "CRAWL",
    "RUN",
    "CROUCH",
    "JUMP"
)

STATE_FIELDS = (
    ("p", 3),
    ("quat", 4),
    ("v_lin", 3),
    ("omega", 3),
    ("q", 12),
    ("qd", 12),
    ("fz", 4)
)

DIAG_NAMES = (
    "walk_active",
    "v_cmd_x",
    "v_cmd_y",
    "wz_cmd",
    "z_des",
    "ground_z",
    "fl_z_des",
    "st_FL",
    "st_FR",
    "st_BL",
    "st_BR",
    "s_lin",
    "s_ang",
    "s_swing",
    "miss_lin",
    "miss_ang",
    "miss_swing",
    "s_max",
    "frozen",
    "wbc_ok",
    "fell_t",
    "yaw",
    "target_yaw",
)

#that qddti sounds gibberish but it's supposed to define a compiled layout for when python value is going to be packed and unpacked on c.
#something like that.
STATE_HEAD = struct.Struct("<cQddIi")
CMD_HEAD = struct.Struct("<cQddi")

def _code(name):
    return STATE_NAMES.index(name) if name in STATE_NAMES else -1

def pack_hello(cfg):
    plain = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()}
    return KIND_HELLO + json.dumps(plain).encode()

def unpack_hello(buf):
    return json.loads(buf[1:].decode())

def pack_state(seq, t, t_wall, st):
    body = np.concatenate([np.asarray(st[k], dtype="<f8").ravel() for k, _ in STATE_FIELDS])
    return STATE_HEAD.pack(KIND_STATE, seq, t, t_wall, st["req_id"], _code(st["req"])) + body.tobytes()

def unpack_state(buf):
    _, seq, t, t_wall, req_id, req_code = STATE_HEAD.unpack_from(buf)
    body = np.frombuffer(buf, dtype="<f8", offset=STATE_HEAD.size)
    st, i = {}, 0

    for k, n in STATE_FIELDS:
        st[k] = body[i:i + n].copy()
        i += n

    st["req_id"] = req_id
    st["req"] = STATE_NAMES[req_code] if req_code >= 0 else None

    return seq, t, t_wall, st

def pack_command(answered_seq, t_wall_echo, compute_s, state_name, tau, diag):
    body = np.concatenate([np.asarray(tau, dtype="<f8"), [diag[k] for k in DIAG_NAMES]]).astype("<f8")
    return CMD_HEAD.pack(KIND_CMD, answered_seq, t_wall_echo, compute_s, _code(state_name)) + body.tobytes()

def unpack_command(buf):
    _, answered_seq, t_wall_echo, compute_s, state_code = CMD_HEAD.unpack_from(buf)

    body = np.frombuffer(buf, dtype="<f8", offset=CMD_HEAD.size)
    tau = body[0:12].copy()
    diag = {k: float(v) for k, v in zip(DIAG_NAMES, body[12:])}

    state_name = STATE_NAMES[state_code] if state_code >= 0 else "?"
    return answered_seq, t_wall_echo, compute_s, state_name, tau, diag

