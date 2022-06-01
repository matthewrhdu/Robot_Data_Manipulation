import numpy as np
from open3d.cpu.pybind.geometry import OrientedBoundingBox
from Algorithms_Builtins import *


def main(filename):
    data_points = read_data(filename)
    pcd_points = run_ransac(data_points, 0.02)

    filtered_clusters = run_dbscan(pcd_points)

    n = 1
    for points in filtered_clusters:
        bounding_box = draw_bounding_box(points, OrientedBoundingBox)
        box_points = np.array(bounding_box.get_box_points())
        box_center = np.asarray(bounding_box.get_center())

        print(len(points))
        print(box_points, box_center)

        n += 1


if __name__ == "__main__":
    main('Data/img0.ply')


