import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# frame = None
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit(1)
#
# while True:
#     # Capture frame-by-frame
#     ret, img = cap.read()
#
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#
#     # Display the resulting frame
#     cv2.imshow('frame', img)
#     key = cv2.waitKey(1)
#
#     if key == ord('s'):
#         frame = img
#         cv2.imwrite("saved.png", img)
#         break
#
#     if key == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
#
# if frame is None:
#     print("Nothing saved")
#     exit(1)

frame = cv2.imread("saved.png")

# plt.figure()
# plt.imshow(frame)
# plt.show()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

cameraMatrix = np.array([[676.8134765625, 0, 482.47442626953125],
                         [0, 677.247314453125, 276.1454772949219],
                         [0, 0, 1.]])

distCoeffs = np.array([0, 0, 0, 0, 0])

rvecs, tvecs = np.ndarray([]), np.ndarray([])
aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs)

plt.imshow(frame_markers)
for i in range(len(ids)):
    c = corners[i][0]
    plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))

    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);

plt.legend()
plt.show()
