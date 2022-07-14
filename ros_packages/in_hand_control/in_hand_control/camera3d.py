import rclpy
from in_hand_control.device import Device
from services.srv import DoStuff
import cv2 as cv
import in_hand_control.Cameras.PointCloudCamera as PCD_Cam
    

class Camera(Device):
    def __init__(self):
        super().__init__('camera3d', 'camera_command', DoStuff)

    def run(self, request, response):
        self.get_logger().info("Received Request")
        if request.stuff == 0:
            self.get_logger().info("Click")
            PCD_Cam.get_point_cloud(0, "image.npy", False)
            response.status = 0
        else:
            response.status = 1
        
        return response


def main(args=None):
    rclpy.init(args=args)

    camera = Camera()
    try:
        rclpy.spin(camera)
    except KeyboardInterrupt:
        pass

    camera.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()