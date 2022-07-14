"""
Created by: Matthew Du
Date: 16-06-2022 11:28 AM

This is a camera module for the in hand manipulation
"""

# import the necessary packages
import cv2 as cv
import numpy as np
import os
import re


def main():
	cap = cv.VideoCapture(0)
	if not cap.isOpened():
		print("Cannot open camera")
		exit(1)

	while True:
		# Capture frame-by-frame
		ret, img = cap.read()

		# if frame is read correctly ret is True
		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break

		# draw_img = cv.flip(img, 0)
		draw_img = np.copy(img)

		img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
		img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
		_, img_gray = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
		img_gray = cv.Canny(img_gray, 100, 200)

		contours, _ = cv.findContours(img_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
		print(contours)
		cv.drawContours(draw_img, contours, -1, (0, 255, 0), 3)

		mask = np.zeros(img_gray.shape, dtype=np.uint8)
		c = np.array([[it[0][0]] for it in contours], dtype=np.int32)
		cv.drawContours(mask, [c], 0, 1, -1)
		# pts = cv2.findNonZero(mask)
		pixel_points = cv.bitwise_and(img, img, mask=mask)

		# Display the resulting frame
		cv.imshow('frame', draw_img)
		cv.imshow('frame_clone', pixel_points)
		cv.imshow('frame_gray', img_gray)

		key = cv.waitKey(1)
		if key == ord('q') or key == 27:
			break

		if key == ord('s'):
			directory = "."
			file_endings = [int(re.findall('\\d+', filename)[0]) for filename in os.listdir(directory) if filename[-len('.png'):] == '.png']
			if not file_endings:
				largest_ending = -1
			else:
				largest_ending = max(file_endings)

			cv.imwrite(f"{directory}/img{largest_ending + 1}.png", img)
			break

	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()
