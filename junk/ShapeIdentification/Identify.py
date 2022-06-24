# import the necessary packages
import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull


def detect(c):
	# initialize the shape name and approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.05 * peri, True)

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
image = None
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

	draw_img = np.copy(img)
	draw_img = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
	draw_img = cv.medianBlur(draw_img, 3)

	# Initiate ORB detector
	orb = cv.ORB_create()

	# find the keypoints with ORB
	key_points = orb.detect(draw_img, None)

	# compute the descriptors with ORB
	kp, des = orb.compute(img, key_points)

	draw_img = cv.drawKeypoints(draw_img, kp, None, color=(0, 255, 0), flags=0)

	# Display the resulting frame
	cv.imshow('frame', draw_img)
	cv.imshow('frame_gray', img)

	key = cv.waitKey(1)
	if key == ord('q'):
		break

	if key == ord('s'):
		cv.imwrite("../../src/In_hand_manipulation/img_0.png", img)
		frame = kp
		image = img
		break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

frame = [np.array(it.pt) for it in frame]

if frame is None:
	exit(0)

centers = []
clusters = []

corners = [frame[0]] * 4
for point in frame[1:]:

	if point[0] > corners[0][0] and point[1] > corners[0][1]:
		corners[0] = point
	if point[0] > corners[1][0] and point[1] < corners[1][1]:
		corners[1] = point
	if point[0] < corners[2][0] and point[1] > corners[2][1]:
		corners[2] = point
	if point[0] < corners[3][0] and point[1] < corners[3][1]:
		corners[3] = point

print(corners)
for pt in corners:
	draw_pt = np.array([int(pt[0]), int(pt[1])])
	cv.drawMarker(img, draw_pt, (255, 255, 255), thickness=5)

clusters.append(corners)


cv.imshow("img", img)
key = cv.waitKey(0)

