import rclpy
from rclpy.node import Node
import time
import math
import serial
import tkinter as tk
from tkinter import ttk
import threading
from quins_jetson.zero_point_init import get_angle_relative 

# PORT = '/dev/ttyCH341USB0'
# PORT = '/dev/ttyUSB0'
PORT = '/dev/ttyUSB1'
HOST_ID = 253

# NOTE: Default PID Values
KP = 1.0
KD = 1.5
KI = 0.5
VELOCITY = 0.0
TORQUE = 0.0

# NOTE: Clamp PID Values
P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -44.0, 44.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0
T_MIN, T_MAX = -17.0, 17.0
I_MIN, I_MAX = -5.0, 5.0

# NOTE: Motor Object
class MotorData():
    def __init__(self, id):
        # NOTE: Main Variable 
        self.motor_id = id
        self.angle = 0.0
        self.enabled = False
        self.last_fault = None
        self.zero_point = 0.0
        self.prev_zero_point = 0.0

        # NOTE: PID Input Value
        self.kp = KP
        self.kd = KD
        self.ki = KI 
        self.velocity = VELOCITY
        self.torque = TORQUE
        self.error = 0.0

        # NOTE: Error Variable 
        self.last_error = 0.0
        self.actual_velocity = 0.0
        self.integral_error = 0.0
        self.last_cmd_time: float | None = None


        # NOTE: GUI Input
        self.angle_input = None
        self.zeropoint_input = None
        self.kp_input = None
        self.kd_input = None
        self.velocity_input = None
        self.torque_input = None


class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        self.port = PORT
        self.baudrate = 921600
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)

        self.host_id = HOST_ID
        self.motors = []
        self.stopped = False

        self.motors.append(MotorData(127))
        # self.motors.append(MotorData(4))
        # self.motors.append(MotorData(5))

        for m in self.motors:
            self.enable_motor(m)
            time.sleep(0.2)

        self.get_logger().info("Program Initialized. Ready for Tuning / Control.")
        # self.timer = self.create_timer(0.005, self.timer_callback)

        for motor in self.motors:
            self.send_command(motor)

    # NOTE: decode from float to unsigned integer
    def float_to_uint(self, x, x_min, x_max, bits):
        span = x_max - x_min
        offset = x_min
        if x > x_max:
            x = x_max
        elif x < x_min:
            x = x_min
        return int(((x - offset) * ((1 << bits) - 1)) / span)

    # NOTE: encode from unsigned integer to float 
    def uint_to_float(self, x_int, x_min, x_max, bits):
        span = x_max - x_min
        return (x_int * span / ((1 << bits) - 1)) + x_min

    # NOTE: Raw send Message to Motor
    def send_can_packet(self, comm_type, target_id, data16, data):
        real_id = (comm_type << 24) | ((data16 & 0xFFFF) << 8) | (target_id & 0xFF)
        encoded_id = (real_id << 3) | 0x04

        payload = bytearray([
            (encoded_id >> 24) & 0xFF,
            (encoded_id >> 16) & 0xFF,
            (encoded_id >> 8) & 0xFF,
            encoded_id & 0xFF,
            8,
        ]) + data

        frame = bytearray([0x41, 0x54]) + payload + bytearray([0x0D, 0x0A])
        self.ser.reset_input_buffer()
        self.ser.write(frame)

        return self.read_response()

    # NOTE:  Raw read Motor Response 
    def decode_response(self, raw):
        encoded_id = int.from_bytes(raw[2:6], 'big')
        real_id = (encoded_id - 0x04) >> 3

        comm_type = (real_id >> 24) & 0x1F
        data16 = (real_id >> 8) & 0xFFFF
        target_id = real_id & 0xFF
        data = raw[7:15]

        result = {'comm_type': comm_type, 'target_id': target_id, 'data16': data16, 'raw_data': data}

        if comm_type == 2:
            # Type 2 feedback: bit8-15 = current CAN ID, bit16-21 = fault, bit22-23 = mode
            can_id = data16 & 0xFF
            fault_bits = (data16 >> 8) & 0x3F
            mode = (data16 >> 14) & 0x03
            result.update({
                'can_id': can_id,
                'fault_bits': fault_bits,
                'mode': mode,
                'angle_raw': int.from_bytes(data[0:2], 'big'),
                'velocity_raw': int.from_bytes(data[2:4], 'big'),
                'torque_raw': int.from_bytes(data[4:6], 'big'),
                'temp_raw': int.from_bytes(data[6:8], 'big'),
            })

        return result

    # NOTE: Enable Motor (for Startup)
    def enable_motor(self, m: MotorData):
        resp = self.send_can_packet(3, m.motor_id, self.host_id, bytearray(8))
        if resp is None:
            self.get_logger().warn(f"Motor {m.motor_id}: no response to Enable frame")
            m.enabled = False
            return
        if resp.get('fault_bits'):
            self.get_logger().warn(f"Motor {m.motor_id}: fault bits {resp['fault_bits']:#x} on enable")
        m.enabled = True
        self.get_logger().info(f"Motor {m.motor_id}: enabled, mode={resp.get('mode')}")

    # NOTE: Disable Motor (on Exit)
    def stop_motor(self, m: MotorData):
        self.send_can_packet(4, m.motor_id, self.host_id, bytearray(8))
        m.enabled = False

    # NOTE: Set Current Angle into its Zero Point
    def set_zero_position(self, m: MotorData):
        data = bytearray(8)
        data[0] = 1
        resp = self.send_can_packet(6, m.motor_id, self.host_id, data)
        if resp is None:
            self.get_logger().warn(f"Motor {m.motor_id}: no response to Set Zero frame")
            return
        self.get_logger().info(f"Motor {m.motor_id}: zero position set")

    # NOTE: Send Input Command
    def send_command(self, target_motor: MotorData, doPing = False):
        now = time.perf_counter()
        if target_motor.last_cmd_time is None:
            dt = 0.0
        else:
            dt = now - target_motor.last_cmd_time
            if dt <= 0.0 or dt > 0.1:
                dt = 0.01
        target_motor.last_cmd_time = now

        pi2 = math.pi * 2
        angle_by_degree = (target_motor.angle / 360) * pi2 
        clamped_angle = min(pi2, max(-pi2, angle_by_degree))

        target_motor.integral_error += target_motor.error * dt
        target_motor.integral_error = max(I_MIN, min(target_motor.integral_error, I_MAX))
        target_motor.torque = target_motor.ki * target_motor.integral_error

        p_int = self.float_to_uint(clamped_angle, P_MIN, P_MAX, 16)
        v_int = self.float_to_uint(target_motor.velocity, V_MIN, V_MAX, 16)
        kp_int = self.float_to_uint(target_motor.kp, KP_MIN, KP_MAX, 16)
        kd_int = self.float_to_uint(target_motor.kd, KD_MIN, KD_MAX, 16)
        t_int = self.float_to_uint(target_motor.torque, T_MIN, T_MAX, 16)

        data = bytearray(8)
        data[0] = p_int >> 8
        data[1] = p_int & 0xFF
        data[2] = v_int >> 8
        data[3] = v_int & 0xFF
        data[4] = kp_int >> 8
        data[5] = kp_int & 0xFF
        data[6] = kd_int >> 8
        data[7] = kd_int & 0xFF

        resp = self.send_can_packet(1, target_motor.motor_id, t_int, data)

        if resp and 'angle_raw' in resp:
            actual_angle_rad = self.uint_to_float(resp['angle_raw'], P_MIN, P_MAX, 16)

            target_motor.actual_velocity = self.uint_to_float(resp['velocity_raw'], V_MIN, V_MAX, 16)
            target_motor.error = clamped_angle - actual_angle_rad
    
            if doPing == True:
                target_motor.last_error = target_motor.error 
                print(f"Inputted: {target_motor.angle}, Target rad: {clamped_angle:.4f}, Actual rad: {actual_angle_rad:.4f}")
                print(f"Error rad: {target_motor.last_error:.4f}")

        if resp is None:
            self.get_logger().warn(f"Motor {target_motor.motor_id}: no feedback frame received")
            target_motor.enabled = False
            return

        fault_bits = resp.get('fault_bits', 0)
        if fault_bits:
            if target_motor.last_fault != fault_bits:
                self.get_logger().warn(f"Motor {target_motor.motor_id}: fault bits {fault_bits:#x}")
            target_motor.last_fault = fault_bits

        mode = resp.get('mode')
        if mode == 0:  # Reset mode
            self.get_logger().warn(f"Motor {target_motor.motor_id}: dropped to RESET mode, re-enabling")
            self.enable_motor(target_motor)

    # NOTE: Read Debug Return Message
    def read_response(self):
        # Response frame: 41 54 [4-byte encoded id] [DLC] [8 data bytes] 0d 0a = 17 bytes
        raw = self.ser.read(17)
        if len(raw) != 17 or raw[0:2] != b'\x41\x54' or raw[-2:] != b'\x0d\x0a':
            return None
        return self.decode_response(raw)

    # NOTE: On Program End
    def destroy_node(self):
        for m in self.motors:
            try:
                self.stop_motor(m)
            except Exception as e:
                self.get_logger().error(f"Motor {m.motor_id}: failed to send stop on shutdown: {e}")
        super().destroy_node()

    # NOTE: Move Motor per Step Helper Function
    def move_motor(self, motors, targets):
        MAX_STEP_DEG = 5.0
        SETTLE_VELOCITY_RAD = 0.5  # Threshold for "stopped moving"
        SETTLE_ERROR_RAD = 0.05    # Threshold for "reached target"
        TIMEOUT = 2.0              # Max seconds to wait before giving up

        for motor in motors:
            motor.integral_error = 0.0

        start_time = time.time()
        while time.time() - start_time < TIMEOUT:
            all_settled = True
            self.stopped = False
            
            for motor, target in zip(motors, targets):
                diff = target - motor.angle
                step = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, diff))
                motor.angle += step
                self.send_command(motor)

                is_moving = abs(motor.actual_velocity) > SETTLE_VELOCITY_RAD
                is_far = abs(motor.error) > SETTLE_ERROR_RAD
                
                if is_moving or is_far:
                    all_settled = False

            if all_settled:
                print("Movement settled.")
                self.stopped = True
                break
                
            time.sleep(0.01)
        
        for motor in motors:
            self.send_command(motor, doPing=True)

def main(args=None):
    rclpy.init(args=args)
    node = MotorController()

    root = tk.Tk()
    root.title("Motor Tuner")
    root.geometry("500x480")

    is_simulating = False

    # NOTE: Applying Angle
    def apply_angle(event=None):
        try:
            targets = [float(m.angle_input.get()) for m in node.motors]
        except ValueError:
            return

        def send_cmd():
            node.move_motor(node.motors, targets)

        threading.Thread(target=send_cmd, daemon=True).start()

    # NOTE: Applying Angle (Shortcut)
    def set_shortcut_angle(motor, angle):
        try:
            targets = [angle]
            motors = [motor]
        except ValueError:
            return

        def send_cmd():
            node.move_motor(motors, targets)

        threading.Thread(target=send_cmd, daemon=True).start()

    # NOTE: Applying Zero Point Function 
    def apply_zero_point(event=None):
        try:
            for motor in node.motors:
                motor.zero_point = float(motor.zeropoint_input.get())
                motor.angle = get_angle_relative(motor.prev_zero_point, motor.zero_point)
        except ValueError:
            return

        def send_both():
            for motor in node.motors:
                if motor.zero_point == motor.prev_zero_point:
                    continue

                node.send_command(motor)
                time.sleep(0.5)

                node.set_zero_position(motor)
                time.sleep(0.5)

                motor.angle = 0
                motor.prev_zero_point = motor.zero_point
                motor.zero_point = 0

        threading.Thread(target=send_both, daemon=True).start()

    # NOTE: Set Current Angle into new Zero Point
    def set_current_zp(id):
        def send_cmd(motor):
            node.set_zero_position(motor)

        threading.Thread(target=lambda: send_cmd(node.motors[id]), daemon=True).start()

    # NOTE: Apply New Param Settings 
    def apply_setting(id):
        try:
            node.motors[id].kp = float(node.motors[id].kp_input.get())
            node.motors[id].kd = float(node.motors[id].kd_input.get())
            node.motors[id].velocity = float(node.motors[id].velocity_input.get())
            # node.motors[id].torque = float(node.motors[id].torque_input.get())
        except ValueError:
            return

    # NOTE: Move Motor a bit (hence jog)
    def jog_motor(id, isRight):
        newAngle = min(360, max(-360, node.motors[id].angle+10 if isRight==True else node.motors[id].angle-10))
        node.motors[id].angle = newAngle

        def send_cmd(d):
            node.send_command(node.motors[d])

        threading.Thread(target=lambda: send_cmd(id), daemon=True).start()

    # NOTE: Simulate Walking Motion
    # WARN: please use this function when testing 1 motor only.
    def simulate_walking(event=None):
        nonlocal is_simulating
        is_simulating = not is_simulating 

        if not is_simulating:
            return

        def walk_sim():
            for m in node.motors:
                m.kd = 4.0
                m.kd = 0.25 

            time.sleep(0.05)
            while is_simulating:
                node.stopped = False
                for m in node.motors:
                    m.angle = 120
                targets = [float(m.angle) for m in node.motors]
                node.move_motor(node.motors, targets)
            
                while not node.stopped:
                    time.sleep(0.01)

                time.sleep(0.01)
                node.stopped = False
                for m in node.motors:
                    m.angle = 250
                targets = [float(m.angle) for m in node.motors]
                node.move_motor(node.motors, targets)

        threading.Thread(target=walk_sim, daemon=True).start()

    # NOTE: GUI CODE

    grid_container = tk.Frame(root)
    grid_container.pack(pady=10, padx=10)

    for id, motor in enumerate(node.motors):
        motor_group = ttk.LabelFrame(grid_container, text=f"Motor {id+1}", padding=15)
        motor_group.grid(row=0, column=id)

        # NOTE: set Angle by Degree Input 
        ttk.Label(motor_group, text=f"Motor {id+1} Angle (rad)").pack()
        motor.angle_input = tk.DoubleVar(value=motor.angle)
        ttk.Entry(motor_group, textvariable=motor.angle_input).pack(fill='x')

        # NOTE: set Zero point via Angle
        ttk.Label(motor_group, text=f"Set Motor {id+1} Zero Point").pack()
        motor.zeropoint_input = tk.DoubleVar(value=motor.zero_point)
        ttk.Entry(motor_group, textvariable=motor.zeropoint_input).pack(fill='x')

        # NOTE: jog Button 
        jm = ttk.Frame(motor_group)
        jm.pack(side="top", pady=10)

        ttk.Label(jm, text=f"Jog Motor {id+1}").pack()
        ttk.Button(jm, text="+", command=lambda idx=id: jog_motor(idx, True)).pack(side="left", padx=5)
        ttk.Button(jm, text="-", command=lambda idx=id: jog_motor(idx, False)).pack(side="left", padx=5)

        # NOTE: set current Angle into new Zero Point 
        tk.Button(motor_group, text="Set Zero Point from Current Angle", command=lambda idx=id: set_current_zp(idx)).pack()

        # NOTE: set Motor Params 
        ttk.Label(motor_group, text=f"Motor {id+1} Settings").pack()

        # kp
        kp = ttk.Frame(motor_group)
        kp.pack(fill='x')
        ttk.Label(kp, text="KP", width=10).pack(side="left", padx=5)
        motor.kp_input = tk.DoubleVar(value=motor.kp)  
        ttk.Entry(kp, textvariable=motor.kp_input).pack(fill='x', side="left")

        # kd
        kd = ttk.Frame(motor_group)
        kd.pack(fill='x')
        ttk.Label(kd, text="kd", width=10).pack(side="left", padx=5)
        motor.kd_input = tk.DoubleVar(value=motor.kd)  
        ttk.Entry(kd, textvariable=motor.kd_input).pack(fill='x')

        # velocity 
        velocity = ttk.Frame(motor_group)
        velocity.pack(fill='x')
        ttk.Label(velocity, text="velocity", width=10).pack(side="left", padx=5)
        motor.velocity_input = tk.DoubleVar(value=motor.velocity)  
        ttk.Entry(velocity, textvariable=motor.velocity_input).pack(fill='x')

        # torque 
        # torque = ttk.Frame(motor_group)
        # torque.pack(fill='x')
        # ttk.Label(torque, text="torque", width=10).pack(side="left", padx=5)
        # motor.torque_input = tk.DoubleVar(value=motor.torque)  
        # ttk.Entry(torque, textvariable=motor.torque_input).pack(fill='x')

        # Submit Param Button
        tk.Button(motor_group, text="Apply Settings", command=lambda idx=id: apply_setting(idx)).pack(fill='x')

        # NOTE: set Angle to fixed Degree (Shortcut)
        ttk.Label(motor_group, text="Angle Shortcut").pack()
        sc = ttk.Frame(motor_group)
        sc.pack(pady=(0, 10))

        tk.Button(sc, text="0°", command=lambda a=0: set_shortcut_angle(motor, a)).grid(row=0, column=0, columnspan=4, sticky="nsew")

        tk.Button(sc, text="90°", command=lambda a=90: set_shortcut_angle(motor, a)).grid(row=1, column=0)
        tk.Button(sc, text="180°", command=lambda a=180: set_shortcut_angle(motor, a)).grid(row=1, column=1)
        tk.Button(sc, text="270°", command=lambda a=270: set_shortcut_angle(motor, a)).grid(row=1, column=2)
        tk.Button(sc, text="360°", command=lambda a=360: set_shortcut_angle(motor, a)).grid(row=1, column=3)
        
        tk.Button(sc, text="-90°", command=lambda a=-90: set_shortcut_angle(motor, a)).grid(row=2, column=0)
        tk.Button(sc, text="-180°", command=lambda a=-180: set_shortcut_angle(motor, a)).grid(row=2, column=1)
        tk.Button(sc, text="-270°", command=lambda a=-270: set_shortcut_angle(motor, a)).grid(row=2, column=2)
        tk.Button(sc, text="-360°", command=lambda a=-360: set_shortcut_angle(motor, a)).grid(row=2, column=3)

    # NOTE: GUI Global Submit Button (Applied to all motors)
    angle_subbutton = tk.Button(grid_container, text="Send Angle", command=apply_angle)
    angle_subbutton.grid(row=3, column=0, columnspan=len(node.motors), sticky="nsew", pady=(20, 5))

    zero_subbutton = tk.Button(grid_container, text="Set Zero Point", command=apply_zero_point)
    zero_subbutton.grid(row=4, column=0, columnspan=len(node.motors), sticky="nsew", pady=5)

    simulate_subbutton = tk.Button(grid_container, text="Simulate Walking", command=simulate_walking)
    simulate_subbutton.grid(row=5, column=0, columnspan=len(node.motors), sticky="nsew", pady=5)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
