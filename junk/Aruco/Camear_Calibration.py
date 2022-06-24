import numpy as np
import cv2, PIL, os
import PIL.Image
from cv2 import aruco
import matplotlib.pyplot as plt

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

board = aruco.CharucoBoard_create(7, 5, 2.5, .8, aruco_dict)


datadir = "calibration_data/"
images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".png") ])
order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
images = images[order]
print(images)

im = PIL.Image.open(images[0])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(im)
# ax.axis('off')
plt.show()


def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")

    all_corners = []
    all_ids = []
    decimator = 0

    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    gray = None
    for img in images:
        print("=> Processing image {0}".format(img))
        frame = cv2.imread(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                all_corners.append(res2[1])
                all_ids.append(res2[2])

        decimator += 1

    if gray is None:
        print("Uh oh...")
        exit(1)

    img_size = gray.shape
    return all_corners, all_ids, img_size


allCorners, allIds, imsize = read_chessboards(images)


def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    camera_matrix_init = np.array([[1000.0,    0., imsize[0] / 2.],
                                   [0.0   , 1000., imsize[1] / 2.],
                                   [0.0   ,    0.,             1.]])

    dist_coeffs_init = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv2.CALIB_RATIONAL_MODEL)

    ret, camera_matrix, distortion_coefficients0,rotation_vectors, translation_vectors,stdDeviationsIntrinsics, std_deviations_extrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=camera_matrix_init,
                      distCoeffs=dist_coeffs_init,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize)
print(mtx)
print(dist)
