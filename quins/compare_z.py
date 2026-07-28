import numpy as np
from stl import mesh 

m = mesh.Mesh.from_file('/home/ulone/ros2_ws/src/quins/meshes/QUAD_BL_LEG_C.stl')
pts = m.vectors.reshape(-1, 3) * 0.1   # apply the same 0.1 scale as the URDF
print("min:", pts.min(axis=0))
print("max:", pts.max(axis=0))
