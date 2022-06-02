import numpy as np
import open3d as o3d
from random import randint

def read_data(filename: str) -> np.ndarray:
    """ Read the .ply file with filename (filename includes .ply) and returns an o3d point cloud"""
    if filename[-len(".ply"):] == ".ply":
        return np.asarray(o3d.io.read_point_cloud(filename).points)
    elif filename[-len(".npy"):] == ".npy":
        return np.load(filename)
    else:
        raise FileNotFoundError


def get_box(box_points: np.ndarray):
    bbl = box_points[0]
    bbr = box_points[1]
    btl = box_points[2]
    fbl = box_points[3]
    ftr = box_points[4]
    ftl = box_points[5]
    fbr = box_points[6]
    btr = box_points[7]
    return [(bbl, bbr), (btl, btr), (bbl, btl), (bbr, btr), (fbl, fbr), (ftl, ftr), (fbl, ftl), (fbr, ftr), (bbl, fbl),
            (btl, ftl), (btr, ftr), (bbr, fbr)]


def visualize_helper(clusters: dict):
    colours = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1]] + [[randint(10, 100) / 100 for s in range(3)] for _ in range(10)]
    q = 0
    vectors = []
    for key in clusters:
        cluster = clusters[key]
        vec = o3d.geometry.PointCloud()
        vec.points = o3d.utility.Vector3dVector(cluster)
        vec.paint_uniform_color(colours[q])
        vectors.append(vec)

        q += 1

    o3d.visualization.draw_geometries(vectors, width=800, height=600)