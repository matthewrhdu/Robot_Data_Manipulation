import rclpy
from in_hand_control.device import Device
from services.srv import DoStuff
import cv2 as cv


class Camera(Device):
    def __init__(self):
        super().__init__('camera', 'camera_command', DoStuff)

    def run(self, request, response):
        self.get_logger().info("Received Request")
        if request.stuff == 0:
            self.get_logger().info("Click")
            response.status = 0
        else:
            response.status = 1
        
        return response

    def take_picture(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().info("Cannot open camera")
            exit(1)

        # Capture frame-by-frame
        ret, img = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            self.get_logger().info("Can't receive frame (stream end?). Exiting ...")
    
        cv.imwrite("image.png", img)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

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