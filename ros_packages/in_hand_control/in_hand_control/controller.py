import rclpy
from rclpy.node import Node
from services.srv import DoStuff, CameraGo, DoThings
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        self.control_server = self.create_service(CameraGo, "in_hand_camera", self.process_robot_request_callback, callback_group=MutuallyExclusiveCallbackGroup())
        
        self.camera_client = self.create_client(DoStuff, "camera_command", callback_group=MutuallyExclusiveCallbackGroup())
        while not self.camera_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        self.processor_client = self.create_client(DoThings, "processor_command", callback_group=MutuallyExclusiveCallbackGroup())
        while not self.processor_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')


    def process_robot_request_callback(self, request, response):
        self.get_logger().info("Command Received")
        
        camera_request = DoStuff.Request()
        camera_request.stuff = request.stuff

        cam_response = self.camera_client.call_async(camera_request)
        while not cam_response.done():
            pass

        self.get_logger().info("Camera Read")

        camera_result = cam_response.result()        
        if camera_result.status == 1:
            response.dx = 0.0
            response.dy = 0.0
            response.dtheta = 0.0
            return response


        processor_request = DoThings.Request()
        processor_request.stuff = request.stuff
        
        processor_response = self.processor_client.call_async(processor_request)
        while not processor_response.done():
            pass
        
        processor_result = processor_response.result()
        response.dx = processor_result.dx
        response.dy = processor_result.dy
        response.dtheta = processor_result.dtheta
        return response

def main(args=None):
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