import rclpy
import numpy as np
import math as m
from rclpy.node import Node

class KinematicsLogic(Node):
    def __init__(self):
        super().__init__('KinematicsLogic')

        self.l1 = 1.2925  
        self.l2 = 1.5005  
        self.l3 = 1.54138

        self.a1 = 0
        self.a2 = 90
        self.a3 = 0

    def denavit_matrix(self, _a, len, d_off, _t):
        theta = m.radians(_t)
        a = m.radians(_a)

        ti = np.array([
            [m.cos(theta), -m.sin(theta), 0, len],
            [m.sin(theta)*m.cos(a), m.cos(theta)*m.cos(a), -m.sin(a), -d_off*m.sin(a)],
            [m.sin(theta)*m.sin(a), m.cos(theta)*m.sin(a),  m.cos(a), d_off*m.cos(a)],
            [0, 0, 0, 1]
        ])

        return ti

    def fk(self, theta1, theta2, theta3):
        self.get_logger().info(f"input are : {theta1}, {theta2}, {theta3}")

        t01 = self.denavit_matrix(self.a1, 0, 0, theta1)
        t12 = self.denavit_matrix(self.a2, self.l1, 0, theta2 - self.a2)
        t23 = self.denavit_matrix(self.a3, self.l2, 0, theta3)
        t34 = np.array([
                [1, 0, 0, self.l3],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        t04 = t01@t12@t23@t34
        t04 = np.round(t04, decimals=4)

        self.get_logger().info(f"Results : \n{t04}")

        return t04

    def ik(self, x, y, z):
        self.get_logger().info(f"input are : {x}, {y}, {z}")
        x4 = x 
        y4 = z 
        z4 = y 

        sqrt_xyl = m.sqrt(max(0.0, x4**2 + y4**2 - self.l1**2))
        theta1 = m.atan2(sqrt_xyl, self.l1) - m.atan2(-y4, x4)

        t3_num = self.l2**2 + self.l3**2 - x4**2 - y4**2 + self.l1**2 - z4**2
        t3_denom = 2 * self.l2 * self.l3
        d = t3_num / t3_denom 
        d = max(-1.0, min(1.0, d))
        theta3 = -( m.pi - m.acos(d) )

        t2_num = self.l3 * m.sin(theta3)
        t2_denum = self.l2 + (self.l3 * m.cos(theta3))
        theta2 = -m.atan2(t2_num, t2_denum) - m.atan2(-z4, sqrt_xyl)
        
        theta1 = m.degrees(theta1)
        theta2 = m.degrees(theta2)
        theta3 = m.degrees(theta3)

        self.get_logger().info(f"Rotation are : {theta1}, {theta2}, {theta3}")
        return theta1, theta2, theta3

    def gait_trajectory(self, phase: float, x_off, z_off, step_len, step_h):
        x = x_off

        if phase < m.pi:
            #STANCE PHASE
            fraction = phase / m.pi
            y = -(step_len / 2.0) + (fraction * step_len)
            z = z_off
        else:
            #SWING PHASE
            fraction = (phase - m.pi) / m.pi
            y = (step_len / 2.0) - (fraction * step_len)
            z = z_off + (step_h * m.sin(fraction * m.pi))

        return x, y, z

    # NOTE: old function that uses manual theta. remove the function on final build
    def gait_angles(self, phase: float, theta2_center, theta2_amplitude, theta3_lift, theta3_stance):
        theta1 = 0.0
        theta2 = theta2_center + theta2_amplitude * m.cos(phase)
        knee_lift = theta3_lift * max(0.0, m.sin(phase))
        theta3 = theta3_stance - knee_lift

        return theta1, theta2, theta3

def main():
    rclpy.init()
    node = KinematicsLogic()

    t04 = node.fk(0, 0, 0)
    x = t04[0, 3]
    y = t04[1, 3]
    z = t04[2, 3]
    node.ik(x, y, z)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
