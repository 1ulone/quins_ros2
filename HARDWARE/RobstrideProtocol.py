import struct

HOST_ID = 253

P_MIN, P_MAX = -12.57, 12.57
V_MIN, V_MAX = -44.0, 44.0
KP_MIN, KP_MAX = 0.0, 100.0
KD_MIN, KD_MAX = 0.0, 5.0
T_MIN, T_MAX = -17.0, 17.0

TYPE_MOTION = 1
TYPE_FEEDBACK = 2
TYPE_ENABLE = 3
TYPE_STOP = 4
TYPE_SET_ZERO = 6

TYPE_PARAM_READ = 17      # read one parameter, the motor answers with a type 17 frame holding the value
TYPE_PARAM_WRITE = 18     # write one parameter (lost on power off)

# Parameter addresses (RS02 English manual, section 4.1.14)
RUN_MODE_INDEX = 0x7005   # uint8: 0 operation control, 1 position mode (PP), 2 velocity, 3 current, 5 position mode (CSP)
LOC_REF_INDEX = 0x7016    # float, rad: position mode target angle
VEL_MAX_INDEX = 0x7024    # float, rad/s: PP speed. The RS02 Chinese manual says 0x7025, check with ParamCheck.py
ACC_SET_INDEX = 0x7025    # float, rad/s^2: PP acceleration. The RS02 Chinese manual says 0x7026, check with ParamCheck.py
RUN_MODE_PP = 1
RUN_MODE_OPERATION = 0


def pack_param_read(motor_id, index, host_id=HOST_ID):
    # bytes 0-1: parameter address (low byte first), bytes 2-7: zero
    data = struct.pack("<HHI", index, 0, 0)
    return make_id(TYPE_PARAM_READ, host_id, motor_id), data


def pack_param_write_float(motor_id, index, value, host_id=HOST_ID):
    # bytes 0-1: parameter address, bytes 2-3: zero, bytes 4-7: float value (low byte first)
    data = struct.pack("<HHf", index, 0, float(value))
    return make_id(TYPE_PARAM_WRITE, host_id, motor_id), data


def pack_param_write_uint8(motor_id, index, value, host_id=HOST_ID):
    # 1-byte parameters (run_mode) go in byte 4, the rest stays zero
    data = struct.pack("<HHB3x", index, 0, int(value))
    return make_id(TYPE_PARAM_WRITE, host_id, motor_id), data


def unpack_param_reply(can_id, data):
    # type 17 reply; returns None for any other frame
    comm_type, data16, target = split_id(can_id)
    if comm_type != TYPE_PARAM_READ or len(data) < 8:
        return None
    return {
        "motor_id": data16 & 0xFF,                            # ID bits 15-8 (same place as in the feedback frame)
        "target": target,                                     # ID bits 7-0, printed in case the motor ID is here instead
        "index": struct.unpack_from("<H", data, 0)[0],
        "float": struct.unpack_from("<f", data, 4)[0],        # value read as a float
        "uint": struct.unpack_from("<I", data, 4)[0],         # same 4 bytes read as an integer (for run_mode, EPScan_time)
        "raw": bytes(data[4:8]),
    }

MODE_RESET, MODE_CALIBRATION, MODE_RUN = 0, 1, 2

def float_to_uint(x, x_min, x_max, bits=16):
    x = min(max(float(x), x_min), x_max)
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))

def uint_to_float(x_int, x_min, x_max, bits=16):
    return x_int * (x_max - x_min) / ((1 << bits) - 1) + x_min

def make_id(comm_type, data16, motor_id):
    return ((comm_type & 0x1F) << 24) | ((data16 & 0xFFFF) << 8) | (motor_id & 0xFF)

def split_id(can_id):
    return (can_id >> 24) & 0x1F, (can_id >> 8) & 0xFFFF, can_id & 0xFF

def pack_motion(motor_id, pos, vel, kp, kd, tau):
    # motor-side units (rad at the output shaft); torque rides in the ID data area
    data = struct.pack(
        ">HHHH",
        float_to_uint(pos, P_MIN, P_MAX),
        float_to_uint(vel, V_MIN, V_MAX),
        float_to_uint(kp, KP_MIN, KP_MAX),
        float_to_uint(kd, KD_MIN, KD_MAX),
    )
    return make_id(TYPE_MOTION, float_to_uint(tau, T_MIN, T_MAX), motor_id), data

def pack_enable(motor_id, host_id=HOST_ID):
    return make_id(TYPE_ENABLE, host_id, motor_id), bytes(8)

def pack_stop(motor_id, clear_fault=False, host_id=HOST_ID):
    data = bytearray(8)
    data[0] = 1 if clear_fault else 0
    return make_id(TYPE_STOP, host_id, motor_id), bytes(data)

def pack_set_zero(motor_id, host_id=HOST_ID):
    data = bytearray(8)
    data[0] = 1
    return make_id(TYPE_SET_ZERO, host_id, motor_id), bytes(data)

def unpack_feedback(can_id, data):
    # type 2 reply: data area bits 0-7 motor ID, 8-13 fault bits, 14-15 mode; None for any other frame
    comm_type, data16, _ = split_id(can_id)
    if comm_type != TYPE_FEEDBACK or len(data) < 8:
        return None
    raw_pos, raw_vel, raw_tau, raw_temp = struct.unpack(">HHHH", bytes(data[0:8]))
    return {
        "motor_id": data16 & 0xFF,
        "fault_bits": (data16 >> 8) & 0x3F,
        "mode": (data16 >> 14) & 0x03,
        "pos": uint_to_float(raw_pos, P_MIN, P_MAX),
        "vel": uint_to_float(raw_vel, V_MIN, V_MAX),
        "tau": uint_to_float(raw_tau, T_MIN, T_MAX),
        "temp": raw_temp / 10.0,
    }
