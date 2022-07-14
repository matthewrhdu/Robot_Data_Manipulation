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
from time import sleep


def main():
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

    acc = []
    i = 1
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # draw_img = cv.flip(img, 0)
        # draw_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        draw_img = np.copy(img)
        cv.drawMarker(draw_img, (draw_img.shape[1] // 2, draw_img.shape[0] // 2), (255, 255, 255), thickness=5)

        pt = img[draw_img.shape[0] // 2][draw_img.shape[1] // 2]

        cv.putText(draw_img, str(pt), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=10)
        if i > 200:
            acc.append(img[draw_img.shape[0] // 2][draw_img.shape[1] // 2])

        if i % 100 == 0:
            print(i - 200)

        # Display the resulting frame
        cv.imshow('frame', img)
        cv.imshow('frame_hsv', draw_img)

        key = cv.waitKey(1)
        if key == ord('q') or key == 27:
            break
        if key == ord('s'):
            print(pt.tolist())

        i += 1
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    print(f"Writing {len(acc)} points")
    arr = np.array(acc)
    np.save("finger_samples.npy", arr)

if __name__ == "__main__":
    main()
