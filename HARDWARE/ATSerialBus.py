import threading
import serial
import time

MIN_FRAME_GAP_S = 0.001 

AT_HEAD = b"AT"
AT_TAIL = b"\r\n"

class ATSerialBus:
    def __init__(self, port, baudrate=921600):
        self.ser = serial.Serial(port, baudrate, timeout=0.01)
        self.ser.reset_input_buffer()
        self.handler = None
        self.rx_frames = 0
        self.bad_bytes = 0
        self.last_send_t = 0.0
        self.running = True
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def set_handler(self, handler):
        self.handler = handler

    def send_frame(self, can_id, data):
        encoded_id = (can_id << 3) | 0x04
        wait = self.last_send_t + MIN_FRAME_GAP_S - time.perf_counter()
        if wait > 0.0:
            time.sleep(wait)      # keep at least MIN_FRAME_GAP_S between frames
        self.ser.write(AT_HEAD + encoded_id.to_bytes(4, "big") + bytes([len(data)]) + bytes(data) + AT_TAIL)
        self.last_send_t = time.perf_counter()

    def _read_loop(self):
        # continuous stream parser: resync on "AT", check the trailer, route by ID (never assume who answered)
        buf = bytearray()
        while self.running:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                continue
            buf += chunk
            while True:
                start = buf.find(AT_HEAD)
                if start < 0:
                    self.bad_bytes += max(len(buf) - 1, 0)
                    del buf[:-1]            # keep the last byte, it may be the "A" of the next header
                    break
                if start > 0:
                    self.bad_bytes += start
                    del buf[:start]
                if len(buf) < 7:            # "AT" + 4-byte ID + DLC
                    break
                dlc = buf[6]
                if dlc > 8:
                    self.bad_bytes += 1
                    del buf[:1]
                    continue
                frame_len = 7 + dlc + 2
                if len(buf) < frame_len:
                    break
                if buf[frame_len - 2:frame_len] != AT_TAIL:
                    self.bad_bytes += 1
                    del buf[:1]
                    continue
                can_id = int.from_bytes(buf[2:6], "big") >> 3
                data = bytes(buf[7:7 + dlc])
                del buf[:frame_len]
                self.rx_frames += 1
                if self.handler is not None:
                    self.handler(can_id, data)

    def close(self):
        self.running = False
        self.reader.join(timeout=0.5)
        self.ser.close()
