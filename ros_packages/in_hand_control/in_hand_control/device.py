from typing import Any
from rclpy.node import Node
from rclpy.action import ActionServer


class Device(Node):
    def __init__(self, name: str, request_topic: str, server_type: Any) -> None:
        super().__init__(name)
        self.srv = ActionServer(self, server_type, request_topic, self.run)
    
    def run(self, msg):
        raise NotImplementedError