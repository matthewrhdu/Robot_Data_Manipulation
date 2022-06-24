import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from open3d.cpu.pybind.geometry import OrientedBoundingBox
from Algorithms_Builtins import *


# Set True to debug
debug = True


def segment_point_cloud(data_source: Union[str, np.ndarray], use_ransac: bool = True, to_save: bool = False) -> None:
    """ Segment a point cloud from a file

    Precondition: Assumes data_source is a 3 x 3 point cloud

    :param to_save: Save to a file if `True`
    :param use_ransac: Use the ransac algorithm if `True`
    :param data_source: The source of the data. If it is a string, it is assumed to be the path to the file. Otherwise,
        it is assumed to be the numpy array of the data
    :return None
    """
    # Get Data
    if isinstance(data_source, np.ndarray):
        data_points = data_source
    else:
        data_points = read_data(data_source)

    if debug:
        print(f"[data_points]: {len(data_points)} points read")

    # Use RANSAC to remove table points
    if use_ransac:
        data_points = run_ransac(data_points, 0.025)

        if debug:
            print(f"[use_ransac]: {len(data_points)} points after RANSAC")

    # Find Clusters
    filtered_clusters, clusters = run_dbscan(data_points, epsilon=0.04, threshold=500)

    if debug:
        for i in range(len(filtered_clusters)):
            print(f"\t[filtered_clusters] {i}: {len(filtered_clusters[i])}")
        visualize_helper(clusters)

    # Getting bounding box
    dataset = []  # For debugging
    for n, points in enumerate(filtered_clusters):
        try:  # Catching RuntimeError that can be raised by bounding box Algorithm
            box_object = draw_bounding_box(points, OrientedBoundingBox())

            box_points = np.array(box_object.get_box_points())
            box_center = np.asarray(box_object.get_center())

            box_corner_points = get_box_pairs(box_points)

            basis1, basis2, basis3 = get_basis_vectors(box_points)

            # Saving points
            if to_save:
                np.save(f"../Data/pov_images/new_images/object_{n}.npy", points)
                np.save(f"../Data/pov_images/new_images/object_{n}_x.npy", points[:, 0])
                np.save(f"../Data/pov_images/new_images/object_{n}_y.npy", points[:, 1])
                np.save(f"../Data/pov_images/new_images/object_{n}_z.npy", points[:, 2])

            if debug:
                # Saving the points to be drawn.
                # By saving the objects as arrays of elements for parameter for plt.plot, I can draw all objects and
                # boxes at once at the end
                dataset.append([points[:, 0], points[:, 1], points[:, 2], 'b.'])
                for box_corner_pairs in box_corner_points:
                    dataset.append([(box_corner_pairs[0][p], box_corner_pairs[1][p]) for p in range(3)] + ['r-'])

                axis1 = np.array([basis1 * (i / 10) + box_center for i in range(-10, 10)])
                axis2 = np.array([basis2 * (i / 10) + box_center for i in range(-10, 10)])
                axis3 = np.array([basis3 * (i / 10) + box_center for i in range(-20, 20)])

                dataset.append([axis1[:, 0], axis1[:, 1], axis1[:, 2], 'm-'])
                dataset.append([axis2[:, 0], axis2[:, 1], axis2[:, 2], 'g-'])
                dataset.append([axis3[:, 0], axis3[:, 1], axis3[:, 2], 'c-'])

            transformation_matrix = np.array([basis1, basis2, basis3])
            print(f"[Transformation Matrix]:\n{transformation_matrix}")

        except RuntimeError:
            if debug:
                print("RuntimeError Detected")

    if debug:
        ax = plt.axes(projection="3d")

        for print_line in dataset:
            # If the line has a colour specified
            if print_line[3] not in ['m-', 'g-', 'c-']:
                ax.plot(*print_line)

            # The only other type of object is the bounding box
            else:
                ax.plot(print_line[0], print_line[1], print_line[2], print_line[3], linewidth=3)

        ax.grid(False)
        ax.set_axis_off()
        plt.show()


if __name__ == "__main__":
    filename = "../Data/pov_images/images2/img_combined.npy"
    segment_point_cloud(filename, use_ransac=False, to_save=True)
