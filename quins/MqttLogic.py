import paho.mqtt.client as mqtt

class MqttLogic():
    def __init__(self):
        self.mqtt_server = "broker.emqx.io"
        self.mqtt_port = 1883

        self.main_topic = "polmanquins/kinematic"
        self.client_id = "ulone_ros_56"

        self.client = mqtt.Client(client_id=self.client_id) 
        self.client.connect(self.mqtt_server, self.mqtt_port, 60)
        self.client.loop_start()

if __name__ == "__main__":
    app = MqttLogic()
