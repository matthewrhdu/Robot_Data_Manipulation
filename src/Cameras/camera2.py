import cv2 as cv
import numpy as np

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
	edges = cv.Canny(draw_img, 100, 200)

	contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
	for i in range(len(contours)):
		color = (0, 255, 0)
		cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

	# Display the resulting frame
	cv.imshow('frame', draw_img)
	cv.imshow('frame_gray', drawing)

	key = cv.waitKey(1)
	if key == ord('q'):
		break

	if key == ord('s'):
		cv.imwrite("../../src/In_hand_manipulation/img_0.png", img)
		frame = contours[0][:, 0]

		image = img
		break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

if frame is None:
	exit(0)

centers = []
clusters = []

peri = cv.arcLength(frame, True)
approx = cv.approxPolyDP(frame, 0.05 * peri, True)

corners = approx[:, 0]
print(corners)
for pt in corners:
	draw_pt = np.array([int(pt[0]), int(pt[1])])
	cv.drawMarker(img, draw_pt, (255, 255, 255), thickness=5)

clusters.append(corners)


cv.imshow("img", img)
key = cv.waitKey(0)
