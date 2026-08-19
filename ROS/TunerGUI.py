import rclpy

from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray 

from ROS.BaseGUI import GUI

class TunerGUI(Node):
    def __init__(self):
        super().__init__('quins_tuner_gui')

        self.state_pub = self.create_publisher(String, '/tuner/state', 10)
        self.params_pub = self.create_publisher(Float64MultiArray, '/tuner/params', 10)
        self.jparams_pub = self.create_publisher(Float64MultiArray, '/tuner/jparams', 10)
        self.phase_offsets_pub = self.create_publisher(Float64MultiArray, '/tuner/phase_offsets', 10)
        self.raw_pub = self.create_publisher(Float64MultiArray, '/tuner/raw', 10)
        self.create_subscription(Float64MultiArray, '/tuner/graph', self.graph_callback, 10)

        callbacks = {
            'state': self.publish_state,
            'phase': self.publish_phase,
            'wt_params': self.publish_wt_params,
            'jt_params': self.publish_jt_params,
            'raw_tune': self.publish_raw_tune,
        }

        self.ui = GUI(callbacks)
        self.ui.setup()

        self.create_timer(0.05, self.ui.update)

    def publish_state(self, val):
        msg = String()
        msg.data = val
        self.state_pub.publish(msg)

    def publish_phase(self, val):
        msg = Float64MultiArray()
        msg.data = val
        self.phase_offsets_pub.publish(msg)

    def publish_wt_params(self, val):
        msg = Float64MultiArray()
        msg.data = val
        self.params_pub.publish(msg)

    def publish_jt_params(self, val):
        msg = Float64MultiArray()
        msg.data = val
        self.jparams_pub.publish(msg)
    
    def publish_raw_tune(self, val):
        msg = Float64MultiArray()
        msg.data = val
        self.raw_pub.publish(msg)

    def graph_callback(self, msg: Float64MultiArray):
        self.ui.update_graph(msg.data[0], msg.data[1], msg.data[2])
        
def main(args=None):
    rclpy.init(args=args)
    node = TunerGUI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
