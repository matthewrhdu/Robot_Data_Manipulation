import numpy as np
import cv2 as cv


cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # cx, cy = frame.shape[0] // 2 + 70, frame.shape[1] // 2
    # cv.rectangle(frame,
    #               (cx + 5, cy + 5),
    #               (cx - 5, cy - 5), (255, 255, 255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv.imwrite("camera.png", frame)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()