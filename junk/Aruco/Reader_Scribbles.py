import numpy as np
import time
import cv2
import cv2.aruco as aruco


def main():
    dist_coefficients = np.array(([[-0.58650416, 0.59103816, -0.00443272, 0.00357844, -0.27203275]]))

    camera_matrix = np.array([[398.12724231,  0.0,            304.35638757],
                              [0.0,           345.38259888,   282.49861858],
                              [0.0,           0.0,            1.0]])

    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        shape = frame.shape[:2]

        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefficients, shape, 0, shape)

        undistorted = cv2.undistort(frame, camera_matrix, dist_coefficients, None, new_camera_mtx)

        x, y, width, height = roi
        undistorted = undistorted[y:y + height, x:x + width]

        frame = undistorted

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        cv2.undistort(frame, camera_matrix, dist_coefficients, None, new_camera_mtx)

        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # If you can't find it, type id
        if ids is not None:
            rotation_vector, translation_vector, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix,
                                                                                     dist_coefficients)
            # Estimate the attitude of each marker and return the values rotation vectors and translation_vector ---
            # different from camera coefficients
            (rotation_vector - translation_vector).any()  # get rid of that nasty numpy value array error

            print("rotation_vector", [np.degrees(item) for item in rotation_vector[0][0]])
            print("translation_vector", translation_vector[0][0])

            for i in range(rotation_vector.shape[0]):
                aruco.drawAxis(frame, camera_matrix, dist_coefficients, rotation_vector[i, :, :], translation_vector[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners)

        # Display result frame
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)

        if key == 27:         # Press esc to exit
            print('esc break...')
            cap.release()
            cv2.destroyAllWindows()
            break

        if key == ord(' '):   # Press the spacebar to save
            # num = num + 1
            # filename = "frames_%s.jpg" % num  # Save an image
            filename = str(time.time())[:10] + ".jpg"
            cv2.imwrite(filename, frame)


if __name__ == "__main__":
    main()
