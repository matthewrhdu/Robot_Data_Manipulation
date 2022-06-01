import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

for n in range(6):
    data = np.load(f"pov_images/img_pov{n}_items2.npy")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([cloud])

