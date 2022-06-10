import cv2 as cv
import sys
import numpy as np


def thresh_callback(canny_output):
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly, bound_rect, centers, radius = [], [], [], []
    for i, cont in enumerate(contours):
        contours_poly.append(cv.approxPolyDP(cont, 3, True))
        bound_rect.append(cv.boundingRect(contours_poly[i]))

        arr = cv.minEnclosingCircle(contours_poly[i])
        centers.append(arr[0])
        radius.append(arr[1])

    y_pos = [it[1] for it in centers]
    min_centers = max(y_pos)
    i = y_pos.index(min_centers)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    color = (0, 255, 0)
    cv.drawContours(drawing, contours_poly, i, color)
    cv.rectangle(drawing,
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
    print(radius[i])
    cv.imshow('Contours', drawing)


img = cv.imread("Hand_object.png")
if img is None:
    sys.exit("Could not read the image.")

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
sensitivity = 100
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_white, upper_white)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img, img, mask=mask)

cv.imshow("Display window", res)
cv.waitKey(0)

img2 = cv.GaussianBlur(res, (5, 5), 5)

cv.imshow("Display window", img2)
cv.waitKey(0)
