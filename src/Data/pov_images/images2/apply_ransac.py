import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from src.point_cloud_segmentation.Algorithms_Builtins import run_ransac


def main():
    for i in range(5):
        data = np.load(f"image{i}_cleaned.npy")
        outs = run_ransac(data, 0.01)
        # outs = data

        out_cloud = o3d.geometry.PointCloud()
        out_cloud.points = o3d.utility.Vector3dVector(outs)
        out_cloud.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([out_cloud])
        np.save(f"image{i}_ransaced_cleaned.npy", outs)


if __name__ == "__main__":
    main()
