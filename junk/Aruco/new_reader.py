import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon


def main():
    # dist_coefficients = np.array(([[-0.58650416, 0.59103816, -0.00443272, 0.00357844, -0.27203275]]))
    #
    # camera_matrix = np.array([[398.12724231,  0.0,            304.35638757],
    #                           [0.0,           345.38259888,   282.49861858],
    #                           [0.0,           0.0,            1.0]])
    # camera_matrix = np.array([[676.8134765625, 0.0, 482.47442626953125],
    #                           [0.0, 677.247314453125, 276.1454772949219],
    #                           [0.0, 0.0, 1.0]])
    #
    # camera_matrix = np.array([[249.41961249, 0.0, 324.33928339],
    #                           [0.0, 249.41961249, 138.39555051],
    #                           [0.0, 0.0, 1.0]])

    camera_matrix = np.array([[654.68470569, 0.0, 309.89837988],
                              [0.0, 654.68470569, 177.32891715],
                              [0.0, 0.0, 1.0]])

    dist_coefficients = np.array(([[0.0, 0.0, 0.0, 0.0, 0.0]]))

    size = 30
    aruco_dict = aruco.custom_dictionary(size, size)
    shape = aruco_dict.bytesList.shape
    new = np.load("aruco_bytes.npy")

    aruco_dict.bytesList = np.empty(shape=shape, dtype=np.uint8)
    aruco_dict.bytesList = new

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # img = cv2.flip(img, 1)

        shape = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, gray = cv2.threshold(gray, 80, 256, cv2.THRESH_BINARY)

        parameters = aruco.DetectorParameters_create()
        cv2.undistort(gray, camera_matrix, dist_coefficients, None, new_camera_mtx)

        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners[0], 0.02, camera_matrix,
                                                                                     dist_coefficients)

            for i in range(rotation_vector.shape[0]):
                aruco.drawAxis(img, camera_matrix, dist_coefficients, rotation_vector[i, :, :],
                               translation_vector[i, :, :], 0.01)
                aruco.drawDetectedMarkers(gray, corners)

            mat = cv2.Rodrigues(rotation_vector)
            r = Rotation.from_matrix(mat[0])
            angle = r.as_euler('xyz', degrees=True)
            cv2.putText(img, f"The angle: {round(angle[2], 2)} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=3)
            print(translation_vector)
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
