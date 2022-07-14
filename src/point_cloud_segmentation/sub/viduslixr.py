import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from src.point_cloud_segmentation.Algorithms_Builtins import run_dbscan

pts = np.load("sub0.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcd], width=800, height=600)