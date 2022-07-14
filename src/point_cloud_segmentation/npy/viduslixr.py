import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.point_cloud_segmentation.Algorithms_Builtins import run_dbscan
pts = np.load("img3.npy")

pts, _ = run_dbscan(pts, 10_000, epsilon=0.001)

ax = plt.axes(projection='3d')
data = pts[0]
for pt in pts:
    if len(pt) < len(data):
        data = pt

plt.plot(data[:, 0], data[:, 1], data[:, 2], 'b.')
plt.show()
