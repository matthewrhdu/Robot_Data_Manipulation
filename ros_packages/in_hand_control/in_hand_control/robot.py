import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from services.action import InitiateCamera
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class Robot(Node):
    """ A robot node. This is a placeholder for the actions of the robotic arm 
    
    === Attributes ===
    :ivar robot_client: the client to connect to the controller.
    """
    robot_client: ActionClient

    def __init__(self) -> None:
        """ Initializer """
        super().__init__('robot')
        self.robot_client = ActionClient(self, InitiateCamera, "in_hand_camera")

    def print_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"going to ({feedback.dx}, {feedback.dy}, {feedback.dtheta})")

    def send_request(self) -> None:
        """ Send a `CameraGo` request to the controller """
        # Create the request
        camera_request = InitiateCamera.Goal()
        camera_request.action_key = 0

        self.robot_client.wait_for_server()

        # Get the response
        future = self.robot_client.send_goal_async(camera_request, feedback_callback=self.print_feedback)
        while not future.done():
            pass
        
        robot_result = future.result()
        future = robot_result.get_result_async()
        while not future.done():
            pass

        response = future.result().result
        self.get_logger().info(f'Result: ({response.dx}, {response.dy}, {response.dtheta})')


def main(args=None):
    """ The main function to run the node """
    rclpy.init(args=args)

    robot = Robot()
    robot.send_request()
    rclpy.spin(robot)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()