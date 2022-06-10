import numpy as np
import cv2 as cv
from typing import List, Union


def clean(lst: List) -> List:
    if len(lst) == 2 and isinstance(lst[0], int) and isinstance(lst[1], int):
        return [lst]
    else:
        cleaned = []
        for it in lst:
            if isinstance(it, np.ndarray):
                cleaned.extend(clean(it.tolist()))
            else:
                cleaned.extend(clean(it))
        return cleaned


cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv.Canny(img_gray, 50, 200)
    _, thresh = cv.threshold(img_gray, 127, 255, 0)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(img, contours, -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv.imshow('frame', img)
    cv.imshow('frame_gray', img_gray)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        pts = clean(contours)
        pts = np.array(pts)

        np.save("objtgb.npy", pts)
        print('saved')

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# scale_percent = 60
# width = int(img.shape[0] * scale_percent / 100)
# height = int(img.shape[1] * scale_percent / 100)
# dim = height, width
#
# img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)