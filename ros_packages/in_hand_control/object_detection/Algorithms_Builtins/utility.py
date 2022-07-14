import numpy as np
import open3d as o3d
from random import randint
from typing import List


def read_data(filename: str) -> np.ndarray:
    """ Read the file with filename (filename includes extensions) and returns a NumPy point cloud

    :raises FileNotFoundError: If the file is not found

    :param filename: The filename to be read
    :return: The numpy array of points
    """
    if filename[-len(".ply"):] == ".ply":
        return np.asarray(o3d.io.read_point_cloud(filename).points)
    elif filename[-len(".npy"):] == ".npy":
        return np.load(filename)
    else:
        raise FileNotFoundError


def visualize_helper(points: List[np.ndarray]) -> None:
    """ A helper function to visualize the cluster

    :param points: The points to be visualized
    :return:
    """
    # Colours to use to visualize with
    colours = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]] \
              + [[randint(10, 100) / 100 for _ in range(3)] for _ in range(10)]

    vectors = []
    for q, cluster in enumerate(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster)
        pcd.paint_uniform_color(colours[q])

        vectors.append(pcd)

    o3d.visualization.draw_geometries(vectors, width=800, height=600)
