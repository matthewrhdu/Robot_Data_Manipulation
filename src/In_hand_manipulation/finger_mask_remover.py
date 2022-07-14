import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)

while True:
	# Take each frame
	_, frame = cap.read()

	# Convert BGR to HSV
	# hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	hsv = frame
	# define range of blue color in HSV
	lower = np.array([98, 76, 65])
	upper = np.array([231, 192, 176])

	# Threshold the HSV image to get only blue colors
	mask = cv.inRange(hsv, lower, upper)
	mask = cv.bitwise_not(mask)

	# Bitwise-AND mask and original image
	res = cv.bitwise_and(frame, frame, mask=mask)
	cv.imshow('frame', frame)
	cv.imshow('mask', mask)
	cv.imshow('res', res)

	k = cv.waitKey(5)
	if k == 27:
		break

cv.destroyAllWindows()
