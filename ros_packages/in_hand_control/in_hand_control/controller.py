from typing import Any
import rclpy
from rclpy.node import Node
from rclpy.client import Client
from services.action import InitiateCamera, ProcessorResponse, RunCamera
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.action import ActionClient, ActionServer
from rclpy.action.client import ClientGoalHandle

def _setup_device_client(topic: str, node: Node, action_type: Any) -> Client:
    """ A helper function to setup the connection to a new device and creating the client.
    
    :param topic: The topic of the node
    :param node: The client node
    :return the client.
    """
    # A seperate callback group to prevent deadlocks. Allows the node to wait for robot connections and for new connections on a seperate thread. See: https://docs.ros.org/en/foxy/How-To-Guides/Using-callback-groups.html
    device_client = ActionClient(node, action_type, topic, callback_group=MutuallyExclusiveCallbackGroup())
    return device_client


class Controller(Node):
    """ The Controller node to process new requests
    
    === Attributes ===
    :ivar control_server: The server to process requests from the server
    :ivar camera_client: The client to send requests for the camera
    :ivar processor_client: The client to send requests for the processor 
    """
    control_server: ActionServer
    camera_client: ActionClient
    processor_client: ActionClient

    def __init__(self) -> None:
        """ Initializer """
        super().__init__('controller')

        # Server to handle requests from the robot arm.
        self.control_server = ActionServer(self, InitiateCamera, "in_hand_camera", self.process_robot_request_callback, callback_group=MutuallyExclusiveCallbackGroup())
        
        self.robot_feedback_channel = None

        ######################### DEVICES CONNECTORS #########################
        self.camera_client = _setup_device_client("camera_command", self, RunCamera)
        self.processor_client = _setup_device_client("processor_command", self, ProcessorResponse)


    def process_robot_request_callback(self, msg):
        """ A callback function for processing requests from the robot 
        
        :param request: the request sent by the robot
        :param response: the response to return
        :return the completed response
        """
        self.get_logger().info("Command Received")
        self.robot_feedback_channel = msg

        result = RunCamera.Result()
        msg.succeed()

        camera_result = self._send_camera_request(msg.request.action_key)      
        if camera_result.exit_status == 1:
            result.exit_status = 1
            return result

        processor_result = self._set_processor_request(msg.request.action_key)
        result.exit_status = processor_result.exit_status

        self.get_logger().info("Done")
        return result

    def handle_feedback(self, feedback_msg):
        processor_feedback = feedback_msg.feedback
        robot_feedback = InitiateCamera.Feedback()
        
        robot_feedback.dx = processor_feedback.dx
        robot_feedback.dy = processor_feedback.dy
        robot_feedback.dtheta = processor_feedback.dtheta
        
        self.robot_feedback_channel.publish_feedback(robot_feedback)

    def _set_processor_request(self, action_key: int):
        """ Sends a request for the processor
        
        :param request: the request to be sent to the processor
        :return the response from the server
        """ 
        processor_request = ProcessorResponse.Goal()
        processor_request.processor_key = action_key
        
        self.processor_client.wait_for_server()

        processor_response = self.processor_client.send_goal_async(processor_request, feedback_callback=self.handle_feedback)
        while not processor_response.done():
            pass
        
        processor_result = processor_response.result()
        future = processor_result.get_result_async()
        while not future.done():
            pass
        
        self.get_logger().info(str(type(future)))
        return future.result().result

    def _send_camera_request(self, action_key: int):
        """ Sends a request for the camera 
        
        :param request: the request to be sent to the camera
        :return the response from the server
        """
        camera_request = RunCamera.Goal()
        camera_request.configuration = action_key

        self.camera_client.wait_for_server()

        camera_response = self.camera_client.send_goal_async(camera_request)
        while not camera_response.done():
            pass

        camera_result = camera_response.result()
        future = camera_result.get_result_async()
        while not future.done():
            pass
        
        self.get_logger().info(str(type(future)))
        return future.result().result

def main(args=None):
    """ The main function to run the controller node """
    rclpy.init(args=args)

    controller = Controller()
    executer = MultiThreadedExecutor()
    executer.add_node(controller)
    try:
        executer.spin()
    except KeyboardInterrupt:
        pass
    
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()