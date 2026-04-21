import rclpy
import numpy as np
import math as m
from rclpy.node import Node


class KinematicsLogic(Node):
    def __init__(self):
        super().__init__('KinematicsLogic')

        self.l1 = 53.0
        self.l2 = 149.35
        self.l3 = 114.0

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

        # self.get_logger().info(f"Results : \n{t04}")

        return t04
    

def main():
    rclpy.init()
    node = KinematicsLogic()

    node.fk(0, 0, 0)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
