import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import open3d as o3d
from random import randint


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

		# img = cv.flip(img, 0)
		contours, img_gray, thresh = preprocess_image(img)

		cv.imshow("img", img)
		# Display the resulting frame
		cv.imshow('frame', thresh)
		cv.imshow('frame_gray', img_gray)

		key = cv.waitKey(1)
		if key == ord('q'):
			# When everything done, release the capture
			cap.release()
			cv.destroyAllWindows()
			return None, None

		if key == ord('s'):
			# cv.imwrite("pic3.png", img)
			return img, contours[0][:, 0]


def preprocess_image(img):
	draw_img = np.copy(img)
	draw_img = mask_finger(draw_img)

	img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
	contours, img_gray, thresh = find_contours(img_gray)
	cv.drawContours(img, contours, -1, (0, 255, 0), 3)

	find_objects(contours, img)
	return contours, img_gray, thresh


def _new_point(point, centers):
	for seen in centers:
		if np.linalg.norm(point - seen) <= 50:
			return False
	return True


def find_objects(contours, img):
	centers = []
	for contour in contours:
		# Reduce the amount of points to search
		peri = cv.arcLength(contour, True)
		approx = cv.approxPolyDP(contour, 0.05 * peri, True)

		if len(approx) == 4:
			corners = get_corners(contour[:, 0], img)

			center, major_axis = find_major_axis(corners)

			if center is not None and _new_point(center, centers):
				centers.append(center)
				start_arr = center + major_axis
				end_arr = center - major_axis
				start = [int(start_arr[0]), int(start_arr[1])]
				end = [int(end_arr[0]), int(end_arr[1])]
				cv.line(img, start, end, (0, 255, 0), thickness=5)


def find_major_axis(corners):
	hull = ConvexHull(corners)
	if hull.area < 500:
		return None, None
	points = hull.points
	relationships = hull.neighbors

	cx = np.mean(hull.points[hull.vertices, 0])
	cy = np.mean(hull.points[hull.vertices, 1])

	center = np.array([cx, cy])

	# Populate the dictionary
	length_to_side_array = {}
	for neighbour_pair in relationships:
		length = np.linalg.norm(points[neighbour_pair[0]] - points[neighbour_pair[1]])
		if length not in length_to_side_array:
			length_to_side_array[length] = []
		length_to_side_array[length].append(points[neighbour_pair[0]] - points[neighbour_pair[1]])

	# In a perfectly rectangular object, there will almost certainly be two of each length. In this case, I am
	# arbitrarily choosing only the first one
	major_axis = length_to_side_array[max(length_to_side_array.keys())][0]

	return center, major_axis


def find_contours(img_gray):
	img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
	_, thresh = cv.threshold(img_gray, 100, 256, cv.THRESH_BINARY)
	img_gray = cv.Canny(thresh, 100, 200)
	contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	return contours, img_gray, thresh


def mask_finger(draw_img):
	lower = np.array([-20, -17, -14])
	upper = np.array([51, 51, 54])
	# lower = np.array([120, 90, 80])
	# upper = np.array([245, 200, 175])

	mask = cv.inRange(draw_img, lower, upper)
	# mask = cv.bitwise_not(mask)

	mask = 255 - mask
	draw_img = cv.bitwise_and(draw_img, draw_img, mask=mask)
	return draw_img


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
	if frame is not None:
		cv.imshow("img", img)
		cv.waitKey()


if __name__ == "__main__":
	main()
