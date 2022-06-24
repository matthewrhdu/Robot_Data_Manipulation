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
	cap = cv.VideoCapture(1)
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

		draw_img = np.copy(img)
		img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
		img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
		img_gray = cv.Canny(img_gray, 100, 200)
		_, thresh = cv.threshold(img_gray, 127, 255, 0)

		contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		cv.drawContours(draw_img, contours, -1, (0, 255, 0), 3)

		# Display the resulting frame
		cv.imshow('frame', draw_img)
		# cv.imshow('frame_gray', img_gray)

		key = cv.waitKey(1)
		if key == ord('q'):
			break

		if key == ord('s'):
			directory = "."
			file_endings = [int(re.findall('\\d+', filename)[0]) for filename in os.listdir(directory) if filename[-len('.png'):] == '.png']
			if not file_endings:
				largest_ending = 0
			else:
				largest_ending = max(file_endings)

			cv.imwrite(f"{directory}/img{largest_ending + 1}.png", img)
			break

	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()
