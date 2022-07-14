import cv2 as cv
import numpy as np


SIZE = 30

img = cv.imread("feng.jpg", cv.IMREAD_GRAYSCALE)
_, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

img = cv.resize(img, (SIZE, SIZE))
new = np.where(img != 0, 255, 0)
# for item in new:
#     print(item)

new = np.uint8(new)

cv.imshow("img", new)
cv.waitKey(0)

write = np.where(img != 0, 1, 0)
write[0] = 1
write[-1] = 1
for row in write:
    row[0] = 1
    row[-1] = 1
np.save("feng.npy", write)
for it in write:
    print(it)

