import rclpy
from in_hand_control.device import Device
from services.action import ProcessorResponse
import cv2 as cv
import cv2.aruco as aruco
import numpy as np


class Processor(Device):
    def __init__(self):
        super().__init__('processor', "processor_command", ProcessorResponse)

    def run(self, msg):
        self.get_logger().info("Received Request")
        
        result = ProcessorResponse.Result()
        if msg.request.processor_key == 0:
            # positions = self.process_initial_position()
            positions = self.temp()
            for id_ in positions:
                translations = positions[id_][0]
                rotations = positions[id_][1]

                response = ProcessorResponse.Feedback()
                response.dx = float(translations[0])
                response.dy = float(translations[1])
                response.dz = float(translations[2])
                response.dtheta = np.degrees(rotations)

                msg.publish_feedback(response)
            result.exit_status = 0
        else:
            result.exit_status = 1
        
        msg.succeed()
        return result

    def temp(self):
        return {0: [np.array([0.0, 0.0, 0.0]), 0.0]}

    def process_initial_position(self):
        camera_matrix = np.array([
            [676.8134765625, 0.0, 482.47442626953125],
            [0.0, 677.247314453125, 276.1454772949219],
            [0.0, 0.0, 1.0]])
        dist_coefficients = np.array(([[0, 0, 0, 0, 0]]))

        frame = cv.imread("image.png")
        shape = frame.shape[:2]

        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        cv.undistort(frame, camera_matrix, dist_coefficients, None, new_camera_mtx)

        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is None:
            raise ValueError

        id_to_positions = {}
        for x in range(len(ids)):
            rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners[x], 0.025, camera_matrix, dist_coefficients)
                
            (rotation_vector - translation_vector).any()  # get rid of that nasty numpy value array error

            id_to_positions[id[x][0]] = [translation_vector[0][0], rotation_vector[0][0]]

        return id_to_positions


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