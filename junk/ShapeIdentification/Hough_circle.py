# import the necessary packages
import cv2 as cv
import numpy as np


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.01 * peri, True)

        # print(approx)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:

            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape, approx


frame = None
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

i = 0
while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
    # img_gray = cv.Canny(img_gray, 100, 200)

    rows = img_gray.shape[0]
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])

            # circle center
            cv.circle(img, center, 1, (255, 0, 0), 3)

            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)

    # Display the resulting frame
    cv.imshow('frame', img)
    cv.imshow('frame_gray', img_gray)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


