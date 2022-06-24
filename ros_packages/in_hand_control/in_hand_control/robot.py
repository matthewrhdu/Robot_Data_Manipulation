from numpy import byte
import rclpy
from rclpy.node import Node
from services.srv import CameraGo


class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        self.robot_client = self.create_client(CameraGo, "in_hand_camera")
        while not self.robot_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.camera_request = CameraGo.Request()

    def send_request(self):
        self.camera_request.stuff = 0
        self.future = self.robot_client.call_async(self.camera_request)


def main(args=None):
    rclpy.init(args=args)

    robot = Robot()
    robot.send_request()

    while rclpy.ok():
        rclpy.spin_once(robot)
        if robot.future.done():
            try:
                response = robot.future.result()
            except Exception as e:
                robot.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                robot.get_logger().info(f'Result: ({response.dx}, {response.dy}, {response.dtheta})')
            break

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()