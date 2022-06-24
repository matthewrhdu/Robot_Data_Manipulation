import rclpy
from in_hand_control.device import Device
from services.srv import DoThings
import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull


class Processor(Device):
    def __init__(self):
        super().__init__('processor', "processor_command", DoThings)

    def run(self, request, response):
        self.get_logger().info("Received Request")
        dx, dy, dtheta = self.process()
        if request.stuff == 0:
            response.dx = dx
            response.dy = dy
            response.dtheta = dtheta
        else:
            response.dx = 0.0
            response.dy = 0.0
            response.dtheta = 0.0
        
        return response

    def process(self):
        # Take each frame
        frame = cv.imread("image.png")
        draw_img = np.copy(frame)
        img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
        img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)

        # _, thresh = cv.threshold(img_gray, 0, 127, 0)
        _, thresh = cv.threshold(img_gray, 127, 255, 0)
        
        img_gray = cv.Canny(thresh, 100, 200)

        contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        obj = contours[0][:, 0]
        for cont in contours:
            if len(cont) < len(obj):
                obj = cont[:, 0]

        hull = ConvexHull(obj)
        cx = np.mean(hull.points[hull.vertices, 0])
        cy = np.mean(hull.points[hull.vertices, 1])

        return float(cx), float(cy), 0.0

def main(args=None):
    rclpy.init(args=args)

    processor = Processor()
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass

    processor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()