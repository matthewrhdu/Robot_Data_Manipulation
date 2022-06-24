import numpy as np
import cv2 as cv
import time


cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

curr = time.time()
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('frame', frame)

    if time.time() - curr >= 6:
        cv.imwrite(f"img_{i}.png", frame)
        i += 1
        print("Captured")
        curr = time.time()

    if i > 15:
        break

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()