from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

fig = plt.figure()
nx = 4
ny = 3
for i in range(1, nx * ny + 1):
    ax = fig.add_subplot(ny, nx, i)
    img = aruco.drawMarker(aruco_dict, i, 700)

    plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
    ax.axis("off")

plt.savefig("markers.pdf")
plt.show()
