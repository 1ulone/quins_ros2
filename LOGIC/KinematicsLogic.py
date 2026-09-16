import numpy as np
import pinocchio as pin

class KinematicsLogic():
    def __init__(self, urdf_path):
        # Load the model with a floating base, aligning with your DynamicsLogic
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        
        self.robot_world = np.array([0, 0, 0])

        # Map your standard leg IDs to the URDF tip link names
        self.leg_map = {
            'FL': 'tl_tip_link',
            'FR': 'tr_tip_link',
            'BL': 'bl_tip_link',
            'BR': 'br_tip_link'
        }
        
        # Map leg IDs to their specific joint names in the URDF
        self.joint_names = {
            'FL': ['tl_shoulder_joint', 'tl_thigh_joint', 'tl_leg_joint'],
            'FR': ['tr_shoulder_joint', 'tr_thigh_joint', 'tr_leg_joint'],
            'BL': ['bl_shoulder_joint', 'bl_thigh_joint', 'bl_leg_joint'],
            'BR': ['br_shoulder_joint', 'br_thigh_joint', 'br_leg_joint']
        }

        self.foot_ids = {leg: self.model.getFrameId(name) for leg, name in self.leg_map.items()}

        # Cache the index maps for fast numerical solving
        self.q_idx = {}
        self.v_idx = {}
        
        for leg, joints in self.joint_names.items():
            q_i = []
            v_i = []
            for j in joints:
                pin_id = self.model.getJointId(j)
                q_i.append(self.model.joints[pin_id].idx_q)
                v_i.append(self.model.joints[pin_id].idx_v)
            self.q_idx[leg] = q_i
            self.v_idx[leg] = v_i

        # A neutral configuration to base our calculations on
        self.q_neutral = pin.neutral(self.model)

    def get_init_pos(self, leg_id):
        """Returns the resting position of the foot relative to the base."""
        pin.forwardKinematics(self.model, self.data, self.q_neutral)
        pin.updateFramePlacements(self.model, self.data)
        
        frame_id = self.foot_ids[leg_id]
        pos = self.data.oMf[frame_id].translation
        return pos[0], pos[1], pos[2]

    def fk(self, leg_id, theta1, theta2, theta3):
        """Forward kinematics using Pinocchio."""
        q = self.q_neutral.copy()
        idx = self.q_idx[leg_id]
        
        # Convert degrees to radians since your old logic expected degrees
        q[idx[0]] = np.radians(theta1)
        q[idx[1]] = np.radians(theta2)
        q[idx[2]] = np.radians(theta3)
        
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        return self.data.oMf[self.foot_ids[leg_id]].homogeneous

    def ik(self, leg_id, x, y, z, knee_dir=1):
        """Numerical Inverse Kinematics using iterative Jacobian pseudo-inverse."""
        frame_id = self.foot_ids[leg_id]
        target_pos = np.array([x, y, z])
        
        q = self.q_neutral.copy()
        idx_q = self.q_idx[leg_id]
        idx_v = self.v_idx[leg_id]
        
        # Seed the solver slightly bent based on knee_dir to avoid singularities
        q[idx_q[1]] = 0.5 
        q[idx_q[2]] = -1.0 * knee_dir
        
        eps = 1e-4
        IT_MAX = 50
        damp = 1e-6
        alpha = 0.5
        
        for i in range(IT_MAX):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            current_pos = self.data.oMf[frame_id].translation
            err = current_pos - target_pos
            
            if np.linalg.norm(err) < eps:
                break
                
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_leg = J[:3, idx_v] 
            
            # Levenberg-Marquardt step
            v_leg = -J_leg.T @ np.linalg.solve(J_leg @ J_leg.T + damp * np.eye(3), err)
            q[idx_q] += v_leg * alpha
            
        return q[idx_q[0]], q[idx_q[1]], q[idx_q[2]]

    def calculate_step(self, leg_id, tx, ty, tz):
        body_x = tx - self.robot_world[0]
        body_y = ty - self.robot_world[1]
        body_z = tz - self.robot_world[2]
        return self.ik(leg_id, body_x, body_y, body_z)

    def get_jacobian(self, leg_id, theta1, theta2, theta3):
        """Returns the 3x3 linear Jacobian for the leg."""
        q = self.q_neutral.copy()
        idx_q = self.q_idx[leg_id]
        idx_v = self.v_idx[leg_id]
        
        q[idx_q[0]] = theta1
        q[idx_q[1]] = theta2
        q[idx_q[2]] = theta3
        
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        frame_id = self.foot_ids[leg_id]
        J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        return J[:3, idx_v]

# import numpy as np
# import math as m
# import pinocchio as pin
# from scipy.linalg import expm
#
# # BODY_LENGTH = 2.3462
# # BODY_WIDTH = 1.3334
# # GRAVITY = 9.81
#
# class KinematicsLogic():
#     def __init__(self, urdf_path):
#
#         self.model = pin.buildModelFromUrdf(urdf_path)
#
#         # Extract exact body offsets dynamically (using Front Left as baseline)
#         fl_shoulder_id = self.model.getJointId("tl_shoulder_joint")
#         fl_shoulder_placement = self.model.jointPlacements[fl_shoulder_id].translation
#
#         # Extract link lengths dynamically from the joint translations
#         tl_thigh_id = self.model.getJointId("tl_thigh_joint")
#         tl_leg_id = self.model.getJointId("tl_leg_joint")
#
#         thigh_placement = self.model.jointPlacements[tl_thigh_id].translation
#         leg_placement = self.model.jointPlacements[tl_leg_id].translation
#
#         self.h_bw = abs(fl_shoulder_placement[0]) 
#         self.h_bl = abs(fl_shoulder_placement[1]) 
#
#         self.l1 = np.linalg.norm(thigh_placement) 
#         self.l2 = np.linalg.norm(leg_placement) 
#
#         tl_tip_id = self.model.getJointId("tl_leg_to_tip")
#         self.l3 = np.linalg.norm(self.model.jointPlacements[tl_tip_id].translation)
#
#         # NOTE: initial end effector position or sumshit
#         self.ee_io = np.zeros((4, 4, 4), dtype=float) 
#
#         # NOTE: jvso grid array for better / cleaner data readability (for me atleast)
#         self.jvso = np.zeros((4, 3, 6)) 
#         self.robot_world = np.array([0, 0, 0])
#
#         # NOTE: the jvso equation, Eq. (9) 
#         #  transforming (S) Screw axis into a (V) screw velocity
#         def jvso_equation(s_ij):
#             t = s_ij[0:3]
#             r = s_ij[3:6]
#
#             v = np.cross(-r, t)
#             w = r 
#
#             return np.concatenate((v, w))
#
#         # NOTE: screw joint the BSni equation, Eq. (8)
#         def screw_joint(j1, posx, posy, posz, dir):
#             match j1:
#                 case 1:
#                     return np.array([posx, posy, posz, 0, dir, 0]) 
#                 case 2:
#                     return np.array([posx, posy, posz, dir, 0, 0]) 
#                 case 3:
#                     return np.array([posx, posy, posz - self.l2, dir, 0, 0]) 
#
#         # NOTE: calculate the Jvso 
#         for n in range(4):
#             # Right legs (FR=0, BR=1) -> +X; Left legs (BL=2, FL=3) -> -X
#             sign_x = 1.0 if n in [0, 1] else -1.0
#             # Front legs (FR=0, FL=3) -> +Y; Back legs (BR=1, BL=2) -> -Y
#             sign_y = 1.0 if n in [0, 3] else -1.0
#
#             pos_x = self.h_bw * sign_x
#             pos_y = self.h_bl * sign_y
#
#             # The shoulder link extends laterally by l1 on the X-axis
#             ee_x = (self.h_bw + self.l1) * sign_x
#             ee_y = pos_y
#
#             self.ee_io[n] = np.array([
#                 [1, 0, 0, ee_x],
#                 [0, 1, 0, ee_y],
#                 [0, 0, 1, -self.l2 - self.l3],
#                 [0, 0, 0, 1]
#             ])
#
#             for i in range(3):
#                 current_x = ee_x if i >= 1 else pos_x
#                 dir = 1.0 if n > 1 else -1.0 
#
#                 bs_ni = screw_joint(i+1, current_x, ee_y, 0.0, dir)
#                 self.jvso[n][i] = jvso_equation(bs_ni)
#
#     def phase_to_index(self, leg_id):
#         match leg_id:
#             case 'FR': return 0
#             case 'BR': return 1
#             case 'BL': return 2
#             case 'FL': return 3
#
#     def get_init_pos(self, leg_id):
#         n = self.phase_to_index(leg_id)
#         return self.ee_io[n][0, 3], self.ee_io[n][1, 3], self.ee_io[n][2, 3]
#
#     # NOTE: -----FORWARD KINEMATIC--------
#     def fk(self, leg_id, theta1, theta2, theta3):
#
#         # NOTE: harcoded to always need input from a degrees.
#         # easier for me to input degrees. the code will always use radians tho
#         t1, t2, t3 = m.radians(theta1), m.radians(theta2), m.radians(theta3)
#         n = self.phase_to_index(leg_id)
#
#         # self.get_logger().info(f"input are : {round(theta1, 4)}, {round(theta2, 4)}, {round(theta3, 4)}")
#
#         def P(jvso, theta):
#             v = jvso[0:3]
#             w = jvso[3:6]
#
#             matrix = np.array([
#                 [    0, -w[2],  w[1], v[0]],
#                 [ w[2],     0, -w[0], v[1]],
#                 [-w[1],  w[0],     0, v[2]],
#                 [    0,     0,     0,    0]
#             ])
#
#             return expm(matrix * theta)
#
#
#         # NOTE: below are the equation to get the end effector position, Eq. (10)
#         ji = P(self.jvso[n][0], t1) @ P(self.jvso[n][1], t2) @ P(self.jvso[n][2], t3) @ self.ee_io[n]
#
#         # self.get_logger().info(f"Results : \n{ji.round(4)}")
#         return ji
#
#     # NOTE: -----INVERSE KINEMATIC--------
#     def ik(self, leg_id, x_r, y_r, z_r, knee_dir=-1):
#         offset_x = self.h_bw if 'R' in leg_id else -self.h_bw
#         offset_y = self.h_bl if 'F' in leg_id else -self.h_bl
#
#         x = x_r - offset_x
#         y = y_r - offset_y
#         z = z_r
#
#         d = self.l1 if 'R' in leg_id else -self.l1
#
#         l_xz = m.sqrt(x**2 + z**2)
#         if l_xz < abs(d):
#             l_xz = abs(d) + 1e-6
#
#         theta1 = m.asin(np.clip(d / l_xz, -1.0, 1.0)) - m.atan2(x, -z)
#         z_prime = -(x * m.sin(theta1) + z * m.cos(theta1))
#
#         L_yz = m.sqrt(y**2 + z_prime**2)
#         max_reach = self.l2 + self.l3 - 0.01
#         min_reach = abs(self.l2 - self.l3) + 0.01
#         L_yz = np.clip(L_yz, min_reach, max_reach)
#
#         c3 = (L_yz**2 - self.l2**2 - self.l3**2) / (2.0 * self.l2 * self.l3)
#         c3 = np.clip(c3, -1.0, 1.0)
#         theta3 = m.acos(c3) * knee_dir
#
#         phi = m.atan2(y, z_prime)
#         psi = m.atan2(self.l3 * m.sin(theta3), self.l2 + self.l3 * m.cos(theta3))
#         theta2 = phi - psi
#
#
#         # self.get_logger().info(f"Input are : {round(x, 4)}, {round(y, 4)}, {round(z, 4)}")
#
#         # x = x + self.h_bl if 'B' in leg_id else x - self.h_bl
#         # z = z - self.h_bw if 'L' in leg_id else z + self.h_bw
#         #
#         # x_a = x
#         # # y_a = -(m.sqrt(y**2 + z**2 - self.l1**2))
#         # y_a = -(m.sqrt(max(0.0, y**2 + z**2 - self.l1**2)))
#         #
#         # # NOTE: Better actual clamping value
#         # margin = 0.02
#         # max_reach = (self.l2 + self.l3) - margin
#         # min_reach = abs(self.l2 - self.l3) + margin
#         # dist = m.sqrt(x_a**2 + y_a**2)
#         # if dist > max_reach:
#         #     scale = max_reach / dist
#         #     x_a *= scale
#         #     y_a *= scale
#         # elif dist < min_reach and dist > 1e-9:
#         #     scale = min_reach / dist
#         #     x_a *= scale
#         #     y_a *= scale
#         #
#         # alpha_beta_denum = m.sqrt(y**2 + z**2)
#         #
#         # alpha_num = m.fabs(z)
#         # alpha_div = max(-1.0, min(1.0, alpha_num / alpha_beta_denum))
#         #
#         # alpha = m.acos(alpha_div)
#         #
#         # beta_num = self.l1
#         # beta_div = max(-1.0, min(1.0, beta_num / alpha_beta_denum))
#         #
#         # beta = m.acos(beta_div)
#         #
#         # phi_num = m.fabs(x_a)
#         # phi_denum = m.sqrt(x_a**2 + y_a**2)
#         # phi_div = max(-1.0, min(1.0, phi_num / phi_denum))
#         #
#         # phi = m.acos(phi_div)
#         #
#         # varphi_num = self.l2**2 + x_a**2 + y_a**2 - self.l3**2
#         # varphi_denum = 2 * self.l2 * m.sqrt(x_a**2 + y_a**2)
#         # varphi_div = max(-1.0, min(1.0, varphi_num / varphi_denum))
#         #
#         # varphi = m.acos(varphi_div)
#         #
#         # # NOTE: z > 0 only accounts if the leg is we're calulcating are the left ones
#         # # same for z < 0 only valid for right legs.
#         # theta1 = alpha - beta if (( 'L' in leg_id and z > 0 ) or ( 'R' in leg_id and z <0 )) else m.pi - alpha - beta
#         #
#         # # NOTE: we'll switch the numerator here. so that results in a 
#         # # the equation is correct but it's reversed. resulting in 3.14 
#         # # which should be 0. (reverse kinda thing. nothing big)
#         # theta3_num = x_a**2 + y_a**2 - self.l2**2 - self.l3**2  
#         # theta3_denum = 2*self.l2*self.l3
#         # theta3_div = max(-1.0, min(1.0, theta3_num / theta3_denum)) 
#         #
#         # theta3 = m.acos(theta3_div) if knee_dir > 0 else -m.acos(theta3_div)
#         #
#         # # NOTE: yeah f readability. something something one line code
#         # theta2 = (m.pi/2 if x_a > 0 else (-m.pi/2)) + (-varphi if theta3 > 0 else varphi) + (-phi if x_a > 0 else phi)
#         #
#         # self.get_logger().info(f"Results are : {round(theta1, 4)}, {round(theta2, 4)}, {round(theta3, 4)}")
#         return theta1, theta2, theta3 
#
#     # NOTE: getting end effector position relative to the body frame
#     def calculate_step(self, leg_id, tx, ty, tz):
#         body_x = tx - self.robot_world[0]
#         body_y = ty - self.robot_world[1]
#         body_z = tz - self.robot_world[2]
#
#         lx = body_x - self.h_bl if 'F' in leg_id else body_x + self.h_bl
#         ly = body_y
#         lz = body_z - self.h_bw if 'R' in leg_id else body_z + self.h_bw
#
#         theta1, theta2, theta3 = self.ik(leg_id, lx, ly, lz)
#         return theta1, theta2, theta3
#
#     # NOTE: Get Jacobian Matrix 
#     def get_jacobian(self, leg_id, theta1, theta2, theta3):
#         n = self.phase_to_index(leg_id)
#
#         def P(jvso, theta):
#             v = jvso[0:3]
#             w = jvso[3:6]
#             matrix = np.array([
#                 [    0, -w[2],  w[1], v[0]],
#                 [ w[2],     0, -w[0], v[1]],
#                 [-w[1],  w[0],     0, v[2]],
#                 [    0,     0,     0,    0]
#             ])
#             return expm(matrix * theta)
#
#         def Ad(T):
#             R = T[0:3, 0:3]
#             p = T[0:3, 3]
#             p_skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
#             Ad_matrix = np.zeros((6, 6))
#             Ad_matrix[0:3, 0:3] = R
#             Ad_matrix[0:3, 3:6] = p_skew @ R
#             Ad_matrix[3:6, 3:6] = R
#             return Ad_matrix
#
#         S1 = self.jvso[n][0]
#         S2 = self.jvso[n][1]
#         S3 = self.jvso[n][2]
#
#         T1 = P(S1, theta1)
#         T2 = P(S2, theta2)
#         T3 = P(S3, theta3)
#
#         J_s = np.zeros((6, 3))
#         J_s[:, 0] = S1
#         J_s[:, 1] = Ad(T1) @ S2
#         J_s[:, 2] = Ad(T1 @ T2) @ S3
#
#         FK = T1 @ T2 @ T3 @ self.ee_io[n]
#         p = FK[0:3, 3]
#
#         p_skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
#
#         J_linear = J_s[0:3, :] - p_skew @ J_s[3:6, :]
#         return J_linear
