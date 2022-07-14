import numpy as np
import open3d as o3d


for n in range(5):
    data = np.load(f"img{n}.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)
