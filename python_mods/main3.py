import numpy as np
import matplotlib.pyplot as plt
from open3d.cpu.pybind.geometry import OrientedBoundingBox
from Algorithms_Builtins import *


def main(filename):
    data_points = read_data(filename)
    print(data_points)

    pcd_points = data_points
    filtered_clusters = run_dbscan(pcd_points)

    n = 1
    dataset = []
    for points in filtered_clusters:
        bounding_box = draw_bounding_box(points, OrientedBoundingBox)
        box_points = np.array(bounding_box.get_box_points())

        box_center = np.asarray(bounding_box.get_center())

        box = get_box(box_points)

        axis1, axis2, axis3 = get_axis_lines(box_points, box_center)

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

    # ax.set_xlim3d(-0.15, 0.15)
    # ax.set_ylim3d(-0.15, 0.15)
    # ax.set_zlim3d(-0.5, -0.2)

    for s in dataset:
        if s[3] not in ['m-', 'g-', 'c-']:
            ax.plot(*s)
        else:
            ax.plot(s[0], s[1], s[2], s[3], linewidth=3)

    ax.grid(False)
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main("combined.npy")
