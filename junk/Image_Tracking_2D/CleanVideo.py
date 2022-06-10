from typing import List, Tuple
import numpy as np
import cv2 as cv


down_width = 300
down_height = 500
down_points = (down_width, down_height)

KERNAL_SIZE = 5


def thresh_callback(canny_output):
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly, bound_rect, centers, radius = [], [], [], []
    for i, cont in enumerate(contours):
        contours_poly.append(cv.approxPolyDP(cont, 3, True))
        bound_rect.append(cv.boundingRect(contours_poly[i]))

        arr = cv.minEnclosingCircle(contours_poly[i])
        centers.append(arr[0])
        radius.append(arr[1])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(centers)):
        color = (0, 255, 0)
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(
            drawing,
            (
                int(bound_rect[i][0]),
                int(bound_rect[i][1])
            ),
            (
                int(bound_rect[i][0] + bound_rect[i][2]),
                int(bound_rect[i][1] + bound_rect[i][3])
            ),
            color, 2)
        # cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    cv.imshow('Contours', drawing)


cap = cv.VideoCapture('sample_video.MOV')
while cap.isOpened():
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img = cv.resize(img, down_points, interpolation=cv.INTER_LINEAR)
    cv.imshow("img", img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    sensitivity = 100
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(img, img, mask=mask)

    img = cv.GaussianBlur(img, (5, 5), 5)
    img = cv.Canny(img, 100, 200)

    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()