import cv2 as cv
import numpy as np


lower = np.uint8([[[255, 191, 215]]])
upper = np.uint8([[[255, 46, 123]]])

print(cv.cvtColor(lower, cv.COLOR_BGR2HSV))
print(cv.cvtColor(upper, cv.COLOR_BGR2HSV))
