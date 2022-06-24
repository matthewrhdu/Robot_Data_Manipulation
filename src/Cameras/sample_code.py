import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull

# Take each frame
frame = cv.imread("image.png")
draw_img = np.copy(frame)
img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
# _, thresh = cv.threshold(img_gray, 0, 127, 0)
_, thresh = cv.threshold(img_gray, 127, 255, 0)
img_gray = cv.Canny(thresh, 100, 200)

contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(draw_img, contours, -1, (0, 255, 0), 3)

# Display the resulting frame
cv.imshow('thresh', thresh)
cv.imshow('frame', draw_img)
cv.waitKey(0)

obj = contours[0][:, 0]
for cont in contours:
    if len(cont) < len(obj):
        obj = cont[:, 0]

hull = ConvexHull(obj)
cx = np.mean(hull.points[hull.vertices, 0])
cy = np.mean(hull.points[hull.vertices, 1])

cv.drawMarker(draw_img, np.array([int(cx), int(cy)]), (255, 255, 255), thickness=5)
cv.imshow('frame', draw_img)
cv.waitKey(0)
