import rclpy
import numpy as np
import math as m
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState
from rclpy.time import Time
from LOGIC.KinematicsLogic import KinematicsLogic

class NametagPublisher(Node):
    def __init__(self):
        super().__init__('dh_end_effector_publisher')
        
        self.joint_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
            
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.tr_angles = {'s': 0.0, 't': 0.0, 'k': 0.0}

        self.l1 = 1.2925
        self.l2 = 1.5005
        self.l3 = 1.5414
        self.a1 = 0
        self.a2 = 90
        self.a3 = 0

    def denavit_matrix(self, _a, length, d_off, _t):
        theta = m.radians(_t)
        a = m.radians(_a)
        ti = np.array([
            [m.cos(theta), -m.sin(theta), 0, length],
            [m.sin(theta)*m.cos(a), m.cos(theta)*m.cos(a), -m.sin(a), -d_off*m.sin(a)],
            [m.sin(theta)*m.sin(a), m.cos(theta)*m.sin(a),  m.cos(a), d_off*m.cos(a)],
            [0, 0, 0, 1]
        ])
        return ti

    def fk(self, theta_sh, theta_th, theta_k):
        t01 = self.denavit_matrix(self.a1, 0, 0, theta_sh)
        t12 = self.denavit_matrix(self.a2, self.l1, 0, theta_th - self.a2)
        t23 = self.denavit_matrix(self.a3, self.l2, 0, theta_k)
        
        # --- THE OFFSET FIX ---
        # Inject your offsets here. 
        # This makes the offset physically rotate WITH the knee joint.
        # Note: These values are local to the DH frame, so 1.0 here 
        # might point in a different direction than the global RViz grid.
        # Tune these numbers until it snaps to the CAD model's foot.
        t34 = np.array([
                [1, 0, 0, self.l3 -0.75],   # X Offset (Length of the leg)
                [0, 1, 0, -0.75],            # Y Offset (Sideways offset)
                [0, 0, 1, 1.0],            # Z Offset (Vertical offset)
                [0, 0, 0, 1]
            ])

        t_dh = t01 @ t12 @ t23 @ t34

        # # --- ALIGNMENT (DH to ROS) ---
        # alignment_matrix = np.array([
        #     [ 0,  0,  1,  0], 
        #     [-1,  0,  0,  0], 
        #     [ 0, -1,  0,  0], 
        #     [ 0,  0,  0,  1]
        # ])

        # --- YOUR Z-ROTATION ---
        angle = m.radians(90) # Change to -90 if it faces backwards
        rot_z = np.array([
            [m.cos(angle), -m.sin(angle), 0, 0],
            [m.sin(angle),  m.cos(angle), 0, 0],
            [0,             0,            1, 0],
            [0,             0,            0, 1]
        ])

        # Multiply the chain. Order is critical.
        t_final = rot_z @ t_dh
        
        # Return pure matrix translation. NO plus or minus here.
        return float(t_final[0, 3]), float(t_final[1, 3]), float(t_final[2, 3])

    def joint_callback(self, msg):
        try:
            if 'tr_shoulder_joint' in msg.name:
                idx_s = msg.name.index('tr_shoulder_joint')
                idx_t = msg.name.index('tr_thigh_joint')
                idx_k = msg.name.index('tr_leg_joint')
                
                self.tr_angles['s'] = m.degrees(msg.position[idx_s])
                self.tr_angles['t'] = m.degrees(msg.position[idx_t])
                self.tr_angles['k'] = m.degrees(msg.position[idx_k])
        except ValueError:
            pass

    def create_text_marker(self, m_id, dh_x, dh_y, dh_z, rx, ry, rz):
        marker = Marker()
        marker.header.frame_id = 'tr_shoulder_link'
        marker.header.stamp = Time().to_msg()
        marker.ns = "dh_readout"
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position the text slightly above the ghost sphere
        marker.pose.position.x = dh_x
        marker.pose.position.y = dh_y
        marker.pose.position.z = dh_z + 0.2
        
        marker.text = f"DH|{rx:.2f}|{ry:.2f}|{rz:.2f}"
        marker.scale.z = 0.15
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        return marker

    def create_ghost_marker(self, m_id, dh_x, dh_y, dh_z):
        marker = Marker()
        marker.header.frame_id = 'tr_shoulder_link'
        marker.header.stamp = Time().to_msg()
        marker.ns = "dh_end_effector"
        marker.id = m_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = dh_x
        marker.pose.position.y = dh_y
        marker.pose.position.z = dh_z
        
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        return marker

    def timer_callback(self):
        msg_array = MarkerArray()

        dh_x, dh_y, dh_z = self.fk(self.tr_angles['s'], self.tr_angles['t'], self.tr_angles['k'])
        kl = KinematicsLogic()
        t04 = kl.fk(m.degrees(self.tr_angles['s']), m.degrees(self.tr_angles['t']), m.degrees(self.tr_angles['k']))
        x = t04[0, 3]
        y = t04[1, 3]
        z = t04[2, 3]
        
        msg_array.markers.append(self.create_ghost_marker(99, dh_x, dh_y, dh_z)) # type: ignore[attr-defined]
        msg_array.markers.append(self.create_text_marker(100, dh_x, dh_y, dh_z, x, y, z)) # type: ignore[attr-defined]

        self.marker_pub.publish(msg_array)

def main(args=None):
    rclpy.init(args=args)
    node = NametagPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
