import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import open3d as o3d


def use_camera():
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

		img = cv.flip(img, 0)
		draw_img = np.copy(img)
		img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
		img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
		_, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
		img_gray = cv.Canny(thresh, 100, 200)

		contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		cv.drawContours(draw_img, contours, -1, (0, 255, 0), 3)

		if len(contours) > 0 and len(contours[0][:, 0]) > 4:
			corners = get_corners(contours[0][:, 0], img)

			hull = ConvexHull(corners)
			points = hull.points
			relationships = hull.neighbors

			cx = np.mean(hull.points[hull.vertices, 0])
			cy = np.mean(hull.points[hull.vertices, 1])

			center = np.array([cx, cy])

			sides = {}
			for neighbour in relationships:
				length = np.linalg.norm(points[neighbour[0]] - points[neighbour[1]])
				if length not in sides:
					sides[length] = []
				sides[length].append(points[neighbour[0]] - points[neighbour[1]])

			major_axis = sides[max(sides.keys())][0]

			axis_line = np.array([major_axis * t / 10 + center for t in range(-10, 10)])
			cv.line(img, (int(axis_line[0][0]), int(axis_line[0][1])), (int(axis_line[-1][0]), int(axis_line[-1][1])),
					(0, 255, 0), thickness=5)

		cv.imshow("img", img)
		# Display the resulting frame
		cv.imshow('frame', thresh)
		cv.imshow('frame_gray', img_gray)

		key = cv.waitKey(1)
		if key == ord('q'):
			break

		if key == ord('s'):
			cv.imwrite("pic3.png", img)
			return img, contours[0][:, 0]

	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()


def use_disk(filename: str, colour: str):
	assert colour == 'b' or colour == 'w'
	img = cv.imread(filename)

	draw_img = np.copy(img)
	img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
	img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
	if colour == 'b':
		_, thresh = cv.threshold(img_gray, 127, 255, 0)
	else:
		_, thresh = cv.threshold(img_gray, 0, 127, 0)
	img_gray = cv.Canny(thresh, 100, 200)

	contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	cv.drawContours(draw_img, contours, -1, (0, 255, 0), 3)
	cv.imshow("img", draw_img)
	cv.waitKey(0)

	return img, contours[0][:, 0]


def get_corners(frame, img):
	peri = cv.arcLength(frame, True)
	approx = cv.approxPolyDP(frame, 0.01 * peri, True)

	corners = approx[:, 0]
	# print(corners)
	for pt in corners:
		draw_pt = np.array([int(pt[0]), int(pt[1])])
		cv.drawMarker(img, draw_pt, (255, 255, 255), thickness=5)

	# cv.imshow("img", img)
	# cv.waitKey(0)
	return corners


def main():
	# img, frame = use_disk("image2.png", 'w')
	img, frame = use_camera()
	if frame is None:
		exit(300)

	corners = get_corners(frame, img)[:4]

	assert len(corners) == 4  # If this is not true, adjust thresholds!!!

	hull = ConvexHull(corners)
	points = hull.points
	relationships = hull.neighbors
	print(points)
	print(relationships)

	cx = np.mean(hull.points[hull.vertices, 0])
	cy = np.mean(hull.points[hull.vertices, 1])

	center = np.array([cx, cy])

	sides = {}
	for neighbour in relationships:
		length = np.linalg.norm(points[neighbour[0]] - points[neighbour[1]])
		if length not in sides:
			sides[length] = []
		sides[length].append(points[neighbour[0]] - points[neighbour[1]])

	print(sides)
	major_axis = sides[max(sides.keys())][0]

	axis_line = np.array([major_axis * t / 10 + center for t in range(-10, 10)])
	cv.line(img, (int(axis_line[0][0]), int(axis_line[0][1])), (int(axis_line[-1][0]), int(axis_line[-1][1])), (0, 255, 0), thickness=5)
	cv.imshow("img", img)
	cv.waitKey()


if __name__ == "__main__":
	main()
