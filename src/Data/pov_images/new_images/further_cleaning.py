import numpy as np
import open3d as o3d
from src.point_cloud_segmentation.Algorithms_Builtins import run_dbscan
from random import randint


def main(n):
    data = np.load(f"object_{n}.npy")
    print(f"num items in object_{n}", len(data))

    out_cloud = o3d.geometry.PointCloud()
    out_cloud.points = o3d.utility.Vector3dVector(data)
    out_cloud.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([out_cloud], width=800, height=600)

    _, cleaned = run_dbscan(data, 500, len(data) / 100_000)
    clouds = []
    colours = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]] + [[randint(0, 100) / 100 for _ in range(3)] for __ in range(10)]
    largest = np.array([])
    for i, item in enumerate(cleaned):
        if len(item) > len(largest):
            largest = item

        in_cloud = o3d.geometry.PointCloud()
        in_cloud.points = o3d.utility.Vector3dVector(item)
        in_cloud.paint_uniform_color(colours[i])

        clouds.append(in_cloud)

    print(len(clouds))
    o3d.visualization.draw_geometries(clouds, width=800, height=600)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(largest)
    cloud.paint_uniform_color((1, 0, 0))

    o3d.visualization.draw_geometries([cloud], width=800, height=600)


if __name__ == "__main__":
    for i in range(5):
        main(i)
