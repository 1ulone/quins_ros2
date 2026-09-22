import os
import sys
import numpy as np
import casadi as cs

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from LOGIC.MpcLogic import WholeBodyMPC
from LOGIC.GaitLogic import LEG_NAMES, JOINT_NAMES

def main():
    print("Initializing MPC Graph...")
    joints_name_list = [j for leg in LEG_NAMES for j in JOINT_NAMES[leg]]
    
    urdf_path = os.path.abspath('urdf/quadruped.urdf')
    mpc = WholeBodyMPC(urdf_path, joints_name_list, n=3, dt=0.02)
    mpc.opti.solver("fatrop", {"fatrop.print_level": 0})

    print("Performing dummy solve to initialize CasADi internal structures...")
    # Create dummy inputs to force the solver to build the internal NLP
    p0 = np.array([0.0, 0.0, 0.35])
    R0 = np.eye(3)
    qj0 = np.zeros(12)
    v0 = np.zeros(18)
    q_nom = np.zeros(12)
    v_base_des = np.zeros(6)
    swing_vz = np.zeros((4, mpc.n))
    stance = np.ones((4, mpc.n))

    try:
        # We don't care if it converges, we just need it to trigger the build
        mpc.solve(p0, R0, qj0, v0, q_nom, v_base_des, swing_vz, stance)
    except Exception:
        pass 

    print("Generating C code (this might take a minute)...")
    cg = cs.CodeGenerator("mpc_nlp.c")
    
    # Now that it's initialized, we can safely extract it
    cg.add(mpc.opti.casadi_solver)
    cg.generate()
    
    print("Export complete: mpc_nlp.c has been generated.")

if __name__ == "__main__":
    main()
