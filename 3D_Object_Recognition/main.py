import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from open3d.cpu.pybind.geometry import OrientedBoundingBox
from Algorithms_Builtins import *


def main(filename):
    data_points = read_data(filename)

    pcd_points = run_ransac(data_points, 0.025)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    o3d.visualization.draw_geometries([pcd], width=1920 // 2, height=1080 // 2)

    filtered_clusters = run_dbscan(pcd_points)

    n = 1
    dataset = []
    for points in filtered_clusters:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], width=1920 // 2, height=1080 // 2)

        bounding_box = draw_bounding_box(points, OrientedBoundingBox)
        box_pts = np.array(bounding_box.get_box_points())

        box_center = np.asarray(bounding_box.get_center())
        box = get_box(box_pts)

        axis1, axis2, axis3 = get_axis_lines(box_pts, box_center)

        print(len(points))

        dataset.append([points[:, 0], points[:, 1], points[:, 2], 'b.'])
        for pair in box:
            dataset.append([(pair[0][0], pair[1][0]), (pair[0][1], pair[1][1]), (pair[0][2], pair[1][2]), 'r-'])

        cent_axis = np.array([axis1(i / 10) for i in range(-10, 10)])
        perp_axis = np.array([axis2(i / 10) for i in range(-10, 10)])
        line_axis = np.array([axis3(i / 10) for i in range(-20, 20)])

        dataset.append([cent_axis[:, 0], cent_axis[:, 1], cent_axis[:, 2], 'm-'])
        dataset.append([perp_axis[:, 0], perp_axis[:, 1], perp_axis[:, 2], 'g-'])
        dataset.append([line_axis[:, 0], line_axis[:, 1], line_axis[:, 2], 'c-'])
        n += 1

    ax = plt.axes(projection="3d")
    for s in dataset:
        if s[3] not in ['m-', 'g-', 'c-']:
            ax.plot(*s)
        else:
            ax.plot(s[0], s[1], s[2], s[3], linewidth=3)

    ax.grid(False)
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main("pov_images/acc.npy")
