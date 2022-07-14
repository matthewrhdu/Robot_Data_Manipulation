import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation


def main():
    camera_matrix = np.array([[654.68470569, 0.0, 309.89837988],
                              [0.0, 654.68470569, 177.32891715],
                              [0.0, 0.0, 1.0]])

    dist_coefficients = np.array(([[0.0, 0.0, 0.0, 0.0, 0.0]]))

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        shape = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 80, 256, cv2.THRESH_BINARY)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        cv2.undistort(gray, camera_matrix, dist_coefficients, None, new_camera_mtx)

        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        pixelpoints = np.zeros(img.shape, np.uint8)
        if ids is not None:
            rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners[0], 0.02, camera_matrix,
                                                                                     dist_coefficients)

            for i in range(rotation_vector.shape[0]):
                aruco.drawAxis(img, camera_matrix, dist_coefficients, rotation_vector[i, :, :],
                               translation_vector[i, :, :], 0.01)
                aruco.drawDetectedMarkers(gray, corners)

            mask = np.zeros(gray.shape, dtype=np.uint8)
            c = np.array([[it] for it in corners[0][0]], dtype=np.int32)
            cv2.drawContours(mask, [c], 0, 1, -1)
            # pts = cv2.findNonZero(mask)
            pixelpoints = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("a", pixelpoints)
        # Display result frame
        cv2.imshow("frame", img)
        cv2.imshow("gray", gray)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





