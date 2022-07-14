from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint

# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html

size = 30
aruco_dict_new = aruco.custom_dictionary(size, size)
shape = aruco_dict_new.bytesList.shape

new = np.load("MBL_Logo.npy")

# np.save("new_tag.npy", new)

# for _ in range(100):
#     new[randint(0, size - 1)][randint(0, size - 1)] = 0

aruco_dict_new.bytesList = np.empty(shape=shape, dtype=np.uint8)
aruco_dict_new.bytesList[0] = aruco.Dictionary_getByteListFromBits(new)

fig = plt.figure()

ax = fig.add_subplot(4, 3, 1)
img = aruco.drawMarker(aruco_dict_new, 0, 700)

plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
ax.axis("off")

np.save("aruco_bytes.npy", aruco_dict_new.bytesList)

plt.savefig("markers.pdf")
plt.show()
