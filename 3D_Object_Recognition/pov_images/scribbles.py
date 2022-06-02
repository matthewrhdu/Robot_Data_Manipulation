import numpy as np
import open3d as o3d

acc = 0
for i in range(5):
    pts = np.load(f"img_pov{i}_filtered.npy")
    acc += len(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)
print(acc)
