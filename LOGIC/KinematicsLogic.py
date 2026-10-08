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
            'FL': 'LF_FOOT',
            'FR': 'RF_FOOT',
            'BL': 'LH_FOOT',
            'BR': 'RH_FOOT'
        }
        
        # Map leg IDs to their specific joint names in the URDF
        self.joint_names = {
            'FL': ['LF_HAA', 'LF_HFE', 'LF_KFE'],
            'FR': ['RF_HAA', 'RF_HFE', 'RF_KFE'],
            'BL': ['LH_HAA', 'LH_HFE', 'LH_KFE'],
            'BR': ['RH_HAA', 'RH_HFE', 'RH_KFE']
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
        q[idx_q[1]] = 0.5 * knee_dir 
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
