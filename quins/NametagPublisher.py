import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
import rclpy.duration

class NametagPublisher(Node):
    def __init__(self):
        super().__init__('nametag_publisher')
        self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Faster timer for smooth movement
        self.timer = self.create_timer(0.05, self.publish_markers) 

    def create_text_marker(self, m_id, frame_id, text, z_offset=0.2):
        marker = Marker()
        marker.header.frame_id = frame_id
        # Set stamp to 0 to bypass time-sync lag
        marker.header.stamp = Time(seconds=0, nanoseconds=0).to_msg()
        marker.ns = "robot_labels"
        marker.id = m_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = float(z_offset)
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0
        marker.text = text
        return marker

    def publish_markers(self):
        msg = MarkerArray()
        try:
            # Time(seconds=0) is the "Latest Available" flag
            latest = rclpy.time.Time(seconds=0, nanoseconds=0)
            timeout = rclpy.duration.Duration(seconds=0, nanoseconds=10000000)

            # 1. Body
            msg.markers.append(self.create_text_marker(0, 'base_link', "BODY", 0.4))

            # 2. TR Shoulder
            # Use 'latest' instead of Time() or self.get_clock().now()
            t_s = self.tf_buffer.lookup_transform('base_link', 'tr_shoulder_link', latest, timeout)
            ps = t_s.transform.translation
            msg.markers.append(self.create_text_marker(1, 'tr_shoulder_link', f"coxa|{ps.x:.2f}|{ps.y:.2f}|{ps.z:.2f}", 0.2))

            # 3. TR Leg Local
            t_l = self.tf_buffer.lookup_transform('tr_shoulder_link', 'tr_tip_link', latest, timeout)
            pl = t_l.transform.translation
            msg.markers.append(self.create_text_marker(2, 'tr_tip_link', f"end_eff|{pl.x:.2f}|{pl.y:.2f}|{pl.z:.2f}", 0.15))

            self.publisher.publish(msg)
        except:
            pass

def main():
    rclpy.init()
    node = NametagPublisher()
    rclpy.spin(node)
    rclpy.shutdown()
