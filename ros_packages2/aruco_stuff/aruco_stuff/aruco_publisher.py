import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import cv2.aruco as aruco


from std_msgs.msg import Float64


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float64, 'topic', 10)

    def run(self):
        camera_matrix = np.array([[654.68470569, 0.0, 309.89837988],
                                  [0.0, 654.68470569, 177.32891715],
                                  [0.0, 0.0, 1.0]])
        
        dist_coefficients = np.array(([[0.0, 0.0, 0.0, 0.0, 0.0]]))

        cap = cv2.VideoCapture(1)
        while True:
            ret, img = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # img = cv2.flip(img, 0)

            shape = img.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            cv2.undistort(img, camera_matrix, dist_coefficients, None, new_camera_mtx)

            corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                for x in range(len(ids)):
                    rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners[x], 0.02, camera_matrix, dist_coefficients)
                    
                    (rotation_vector - translation_vector).any()  # get rid of that nasty numpy value array error

                    msg = Float64()
                    msg.data = np.degrees(float(rotation_vector[0][0][0]))
                    self.publisher_.publish(msg)
                    self.get_logger().info('Publishing: "%s"' % msg.data)

        cap.release()



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()