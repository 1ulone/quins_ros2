import os 
import pinocchio as pin
import numpy as np

class QuadrupedDynamics:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.foot_names = ['tl_tip_link', 'tr_tip_link', 'bl_tip_link', 'br_tip_link']
        self.foot_ids = [self.model.getFrameId(name) for name in self.foot_names]

        print(f"Successfully loaded Pinocchio model with {self.model.nq} configuration variables and {self.model.nv} velocity variables.")

    def compute_dynamics(self, q, v):
        pin.crba(self.model, self.data, q)
        M = self.data.M
        M = np.triu(M) + np.triu(M, 1).T

        pin.nonLinearEffects(self.model, self.data, q, v)
        b = self.data.nle

        return M, b

    def compute_inverse_dynamics(self, q, v, a):
        return pin.rnea(self.model, self.data, q, v, a)

    def get_foot_jacobians(self, q):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        jacobians = []
        for fid in self.foot_ids:
            j = pin.computeFrameJacobian(
                self.model, 
                self.data,
                q,
                fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            jacobians.append(j[:3, :])

        return jacobians

if __name__ == "__main__":
    urdf_file = os.path.join(os.path.dirname(__file__), '../urdf/quadruped.urdf')
    dyn = QuadrupedDynamics(urdf_file)

    q_test = pin.neutral(dyn.model)
    v_test = np.zeros(dyn.model.nv)

    M, b = dyn.compute_dynamics(q_test, v_test)
    print(f"Mass Matrix M(q) shape: {M.shape}")
    print(f"Non-linear effects b(q,v) shape: {b.shape}")

