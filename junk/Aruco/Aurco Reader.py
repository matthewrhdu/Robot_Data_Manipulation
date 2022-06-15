import numpy as np
import cv2
import cv2.aruco as aruco


def main(debug: bool = True):
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

    frame = None
    cap = cv2.VideoCapture(1)
    while True:
        ret, img = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display result frame
        cv2.imshow("frame", img)

        key = cv2.waitKey(1)
        if key == 27:
            break

        if key == ord('s'):
            frame = img
            break

    cap.release()
    cv2.destroyAllWindows()

    shape = frame.shape[:2]

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

    # undistorted = cv2.undistort(frame, camera_matrix, dist_coefficients, None, new_camera_mtx)
    #
    # x, y, width, height = roi
    # undistorted = undistorted[y:y + height, x:x + width]
    #
    # frame = undistorted

    if debug:
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    cv2.undistort(frame, camera_matrix, dist_coefficients, None, new_camera_mtx)

    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        raise ValueError

    rotations = []
    translations = []
    for x in range(len(ids)):
        if ids[x][0] in [1, 2]:
            rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners[x], 0.025, camera_matrix,
                                                                                     dist_coefficients)
            (rotation_vector - translation_vector).any()  # get rid of that nasty numpy value array error

            translations.append(translation_vector[0][0])
            rotations.append(rotation_vector[0][0])

            if debug:
                for i in range(rotation_vector.shape[0]):
                    aruco.drawAxis(frame, camera_matrix, dist_coefficients, rotation_vector[i, :, :],
                                   translation_vector[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, corners)
    print(translations[1] - translations[0])
    print(np.degrees(rotations[1] - rotations[0]))
    cv2.imshow("frame", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    main(True)
