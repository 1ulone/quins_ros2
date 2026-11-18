import os
import sys
import time
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from HARDWARE.ATSerialBus import ATSerialBus
from HARDWARE import RobstrideProtocol as rp

PORT = "/dev/ttyUSB0"
MOTOR_IDS = [21, 22, 23]          # same as LegTestGUI.py
REPLY_WAIT_S = 0.05               # time to wait for each answer before sending the next request

# addresses to read, with what each manual says is stored there
INDICES = {
    0x7005: "run_mode",
    0x7024: "English: vel_max (default 10)",
    0x7025: "English: acc_set (default 10) / Chinese: vel_max",
    0x7026: "English: EPScan_time (default 1) / Chinese: acc_set",
}


def main():
    bus = ATSerialBus(PORT)
    replies = {}                  # (motor_id, index) -> reply
    lock = threading.Lock()

    def on_frame(can_id, data):
        reply = rp.unpack_param_reply(can_id, data)
        if reply is None:
            return                # not a parameter reply
        with lock:
            replies[(reply["motor_id"], reply["index"])] = reply

    bus.set_handler(on_frame)
    try:
        for motor_id in MOTOR_IDS:
            for index in INDICES:
                bus.send_frame(*rp.pack_param_read(motor_id, index))
                time.sleep(REPLY_WAIT_S)
        time.sleep(0.1)           # last answers
    finally:
        bus.close()

    for motor_id in MOTOR_IDS:
        print(f"\nmotor {motor_id}")
        for index, meaning in INDICES.items():
            r = replies.get((motor_id, index))
            if r is None:
                print(f"  {index:#06x}  {meaning:<50} NO REPLY")
            else:
                print(f"  {index:#06x}  {meaning:<50} float {r['float']:10.4f}   uint {r['uint']:<10}   raw {r['raw'].hex(' ')}")

    expected = {(motor_id, index) for motor_id in MOTOR_IDS for index in INDICES}
    for key, r in replies.items():
        if key not in expected:
            print(f"\nunexpected reply: motor field {r['motor_id']}, target field {r['target']}, "
                  f"index {r['index']:#06x}, float {r['float']:.4f}, raw {r['raw'].hex(' ')}")


if __name__ == "__main__":
    main()
