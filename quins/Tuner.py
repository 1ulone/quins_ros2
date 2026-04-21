import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tkinter as tk
import threading
from quins.KinematicsLogic import KinematicsLogic

class Tuner(Node):
    def __init__(self):
        super().__init__('quins_tuner')
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.joint_names = [
            'bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint',
            'br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint',
            'tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint',
            'tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'
        ]

    def send_theta(self, coxa, femur, tibia):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()

        point.positions = [coxa, femur, tibia] * 4

        point.time_from_start = Duration(sec=0, nanosec=500000000)
        msg.points.append(point) # type: ignore
        self.publisher.publish(msg)


import math # Ensure this is imported

def main(args=None):
    rclpy.init(args=args)
    node = Tuner()
    kinematics = KinematicsLogic()

    def update_robot(val):
        s = float(shoulder_slider.get())
        t = float(thigh_slider.get())
        k = float(knee_slider.get())
        
        # readout_label.config(text=f"Copy these to your script:\n[ {s:.2f},  {t:.2f},  {k:.2f} ]")
        
        node.send_theta(s, t, k)

        # --- ADDED: Calculate and update XYZ ---
        t04 = kinematics.fk(math.degrees(s), math.degrees(t), math.degrees(k))
        x = t04[0, 3]
        y = t04[1, 3]
        z = t04[2, 3]
        
        xyz_label.config(text=f"XYZ Position:\n[ X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f} ]")
        # ---------------------------------------

    # Build UI
    root = tk.Tk()
    root.title("Quadruped Tuner")
    # Increased height to fit the new label
    root.geometry("400x350") 

    tk.Label(root, text="Shoulder Angle", font=("Arial", 10, "bold")).pack()
    shoulder_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=update_robot)
    shoulder_slider.pack()

    tk.Label(root, text="Thigh Angle", font=("Arial", 10, "bold")).pack()
    thigh_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=update_robot)
    thigh_slider.pack()

    tk.Label(root, text="Knee Angle", font=("Arial", 10, "bold")).pack()
    knee_slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01, orient="horizontal", length=300, command=update_robot)
    knee_slider.pack()

    # --- ADDED: UI Label for XYZ ---
    xyz_label = tk.Label(root, text="XYZ Position:\n[ X:0.00, Y:0.00, Z:0.00 ]", font=("Courier", 12, "bold"), fg="red")
    xyz_label.pack(pady=10)
    # -------------------------------

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
