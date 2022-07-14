"""
Created by: Matthew Du
Date: 16-06-2022 11:28 AM

This is a camera module for the in hand manipulation
"""

# import the necessary packages
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Capture frame-by-frame
    img = cv.imread("sample2.jpg")

    # draw_img = cv.flip(img, 0)
    draw_img = cv.resize(img, (480, 480))
    img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
    _, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    img_gray = cv.Canny(thresh, 100, 200)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    points = []
    for cont in contours:
        if len(cont) > 100:
            points.append(cont)

            cv.drawContours(draw_img, [cont], -1, (0, 255, 0), 3)

    pcds = [[]] * len(points)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            for c in range(len(points)):
                ret = cv.pointPolygonTest(points[c], (j, i), True)
                if ret > 0:
                    pcds[c].append([j, i])

    pcds = np.array(pcds)
    for pcd in pcds:
        plt.plot(pcd[:, 0], pcd[:, 1], 'r.')
        plt.xlim(0, img_gray.shape[0])
        plt.ylim(0, img_gray.shape[1])
        plt.show()

    # Display the resulting frame
    cv.imshow('frame', draw_img)
    cv.imshow('frame_thresh', thresh)
    cv.imshow('frame_gray', img_gray)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
