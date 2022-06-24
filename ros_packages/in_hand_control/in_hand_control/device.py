from typing import Any
from rclpy.node import Node


class Device(Node):
    def __init__(self, name: str, request_topic: str, server_type: Any) -> None:
        super().__init__(name)
        self.srv = self.create_service(server_type, request_topic, self.run)
    
    def run(self, request, response):
        raise NotImplementedError