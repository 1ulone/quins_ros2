# import rclpy
import numpy as np
import math as m
from scipy.linalg import expm
# from rclpy.node import Node

BODY_WIDTH = 2.5
BODY_LENGTH = 2.5

# class KinematicsLogic(Node):
class KinematicsLogic():
    def __init__(self):
        # super().__init__('KinematicsLogic')

        self.l1 = 1.2925  
        self.l2 = 1.5005  
        self.l3 = 1.54138

        self.h_bw = BODY_WIDTH / 2.0
        self.h_bl = BODY_LENGTH / 2.0

        # NOTE: initial end effector position or sumshit
        self.ee_io = np.zeros((4, 4, 4), dtype=float) 

        # NOTE: jvso grid array for better / cleaner data readability (for me atleast)
        self.jvso = np.zeros((4, 3, 6)) 
        self.robot_world = np.array([0, 0, 0])

        # NOTE: the jvso equation, Eq. (9) 
        #  transforming (S) Screw axis into a (V) screw velocity
        def jvso_equation(s_ij):
            t = s_ij[0:3]
            r = s_ij[3:6]

            v = np.cross(-r, t)
            w = r 

            return np.concatenate((v, w))

        # NOTE: screw joint the BSni equation, Eq. (8)
        def screw_joint(j1, posx, posz, dir):
            match j1:
                case 1:
                    return np.array([posx, 0, posz, dir, 0, 0]) 
                case 2:
                    return np.array([posx, 0, posz, 0, 0, dir]) 
                case 3:
                    return np.array([posx, -self.l2, posz, 0, 0, dir]) 
        
        # NOTE: calculate the Jvso 
        for n in range(4):
            dir = 1 if n>1 else -1
            hbl = self.h_bl if n==0 or n==3 else (-self.h_bl)

            self.ee_io[n] = np.array([
                [1, 0, 0,  hbl],
                [0, 1, 0, -self.l2 - self.l3],
                [0, 0, 1,  (self.h_bw + self.l1)* dir],
                [0, 0, 0,  1]
            ])
            for i in range(3):
                hbw = (self.h_bw+self.l1)*dir if i>=1 else self.h_bw*dir

                bs_ni = screw_joint(i+1, hbl, hbw, dir)
                self.jvso[n][i] = jvso_equation(bs_ni)

    def phase_to_index(self, leg_id):
        match leg_id:
            case 'FR': return 0
            case 'BR': return 1
            case 'BL': return 2
            case 'FL': return 3

    def get_init_pos(self, leg_id):
        n = self.phase_to_index(leg_id)
        return self.ee_io[n][0, 3], self.ee_io[n][1, 3], self.ee_io[n][2, 3]

    # NOTE: -----FORWARD KINEMATIC--------
    def fk(self, leg_id, theta1, theta2, theta3):

        # NOTE: harcoded to always need input from a degrees.
        # easier for me to input degrees. the code will always use radians tho
        theta1 = m.radians(theta1)
        theta2 = m.radians(theta2)
        theta3 = m.radians(theta3)
        n = self.phase_to_index(leg_id)

        # self.get_logger().info(f"input are : {round(theta1, 4)}, {round(theta2, 4)}, {round(theta3, 4)}")

        def P(jvso, theta):
            v = jvso[0:3]
            w = jvso[3:6]
            
            matrix = np.array([
                [    0, -w[2],  w[1], v[0]],
                [ w[2],     0, -w[0], v[1]],
                [-w[1],  w[0],     0, v[2]],
                [    0,     0,     0,    0]
            ])

            return expm(matrix * theta)


        # NOTE: below are the equation to get the end effector position, Eq. (10)
        ji = P(self.jvso[n][0], theta1) @ P(self.jvso[n][1], theta2) @ P(self.jvso[n][2], theta3) @ self.ee_io[n]

        # self.get_logger().info(f"Results : \n{ji.round(4)}")
        return ji

    # NOTE: -----INVERSE KINEMATIC--------
    def ik(self, leg_id, x_r, y_r, z_r, knee_dir=-1):
        x = x_r
        y = y_r
        z = z_r

        # self.get_logger().info(f"Input are : {round(x, 4)}, {round(y, 4)}, {round(z, 4)}")

        x = x + self.h_bl if 'B' in leg_id else x - self.h_bl
        z = z - self.h_bw if 'L' in leg_id else z + self.h_bw

        x_a = x
        # y_a = -(m.sqrt(y**2 + z**2 - self.l1**2))
        y_a = -(m.sqrt(max(0.0, y**2 + z**2 - self.l1**2)))

        # NOTE: Better actual clamping value
        margin = 0.02
        max_reach = (self.l2 + self.l3) - margin
        min_reach = abs(self.l2 - self.l3) + margin
        dist = m.sqrt(x_a**2 + y_a**2)
        if dist > max_reach:
            scale = max_reach / dist
            x_a *= scale
            y_a *= scale
        elif dist < min_reach and dist > 1e-9:
            scale = min_reach / dist
            x_a *= scale
            y_a *= scale

        alpha_beta_denum = m.sqrt(y**2 + z**2)

        alpha_num = m.fabs(z)
        alpha_div = max(-1.0, min(1.0, alpha_num / alpha_beta_denum))

        alpha = m.acos(alpha_div)
        
        beta_num = self.l1
        beta_div = max(-1.0, min(1.0, beta_num / alpha_beta_denum))

        beta = m.acos(beta_div)

        phi_num = m.fabs(x_a)
        phi_denum = m.sqrt(x_a**2 + y_a**2)
        phi_div = max(-1.0, min(1.0, phi_num / phi_denum))

        phi = m.acos(phi_div)

        varphi_num = self.l2**2 + x_a**2 + y_a**2 - self.l3**2
        varphi_denum = 2 * self.l2 * m.sqrt(x_a**2 + y_a**2)
        varphi_div = max(-1.0, min(1.0, varphi_num / varphi_denum))

        varphi = m.acos(varphi_div)

        # NOTE: z > 0 only accounts if the leg is we're calulcating are the left ones
        # same for z < 0 only valid for right legs.
        theta1 = alpha - beta if (( 'L' in leg_id and z > 0 ) or ( 'R' in leg_id and z <0 )) else m.pi - alpha - beta

        # NOTE: we'll switch the numerator here. so that results in a 
        # the equation is correct but it's reversed. resulting in 3.14 
        # which should be 0. (reverse kinda thing. nothing big)
        theta3_num = x_a**2 + y_a**2 - self.l2**2 - self.l3**2  
        theta3_denum = 2*self.l2*self.l3
        theta3_div = max(-1.0, min(1.0, theta3_num / theta3_denum)) 

        theta3 = m.acos(theta3_div) if knee_dir > 0 else -m.acos(theta3_div)
    
        # NOTE: yeah f readability. something something one line code
        theta2 = (m.pi/2 if x_a > 0 else (-m.pi/2)) + (-varphi if theta3 > 0 else varphi) + (-phi if x_a > 0 else phi)

        # self.get_logger().info(f"Results are : {round(theta1, 4)}, {round(theta2, 4)}, {round(theta3, 4)}")
        return theta1, theta2, theta3 

    # NOTE: getting end effector position relative to the body frame
    def calculate_step(self, leg_id, tx, ty, tz):
        body_x = tx - self.robot_world[0]
        body_y = ty - self.robot_world[1]
        body_z = tz - self.robot_world[2]

        lx = body_x - self.h_bl if 'F' in leg_id else body_x + self.h_bl
        ly = body_y
        lz = body_z - self.h_bw if 'R' in leg_id else body_z + self.h_bw

        theta1, theta2, theta3 = self.ik(leg_id, lx, ly, lz)
        return theta1, theta2, theta3

    def get_spatial_inertia(self, model, data, body_id):
        mass = model.body_mass[body_id]
        
        I_local_diag = model.body_inertia[body_id]
        I_local = np.diag(I_local_diag)
        
        R = data.xmat[body_id].reshape(3, 3)
        
        I_world = R @ I_local @ R.T
        
        spatial_I = np.zeros((6, 6))
        spatial_I[0:3, 0:3] = np.eye(3) * mass
        spatial_I[3:6, 3:6] = I_world
        
        return spatial_I

#
# def main():
#     rclpy.init()
#     node = KinematicsLogic()
#
#     leg_id = 'BR'
#     fk = node.fk(leg_id, 180.0, 10.0, -100.0)
#     x = fk[0, 3]
#     y = fk[1, 3]
#     z = fk[2, 3]
#     node.ik(leg_id, x, y, z)
#
#     rclpy.shutdown()

#
# if __name__ == '__main__':
#     main()
