import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from src.point_cloud_segmentation.Algorithms_Builtins import run_ransac


def main():
    data = np.load(f"../images2/image0.npy")

    out_cloud = o3d.geometry.PointCloud()
    out_cloud.points = o3d.utility.Vector3dVector(data)
    out_cloud.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([out_cloud])


if __name__ == "__main__":
    main()
