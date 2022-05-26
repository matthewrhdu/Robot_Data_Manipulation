import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from open3d.cpu.pybind.geometry import OrientedBoundingBox, AxisAlignedBoundingBox
from typing import Callable, Union


def read_data(filename: str) -> np.ndarray:
    """ Read the .ply file with filename (filename includes .ply) and returns an o3d point cloud"""
    return np.asarray(o3d.io.read_point_cloud(filename).points)


def run_ransac(pcd: np.ndarray, threshold: float):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=100)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    return np.array(outlier_cloud.points)


def run_dbscan(point_cloud: np.ndarray):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    mask = z <= min(z) + (max(z) - min(z)) / 2
    matrix = np.column_stack((x[mask], y[mask], z[mask]))

    # matrix = np.column_stack((x, y, z))
    min_to_be_classified_as_a_cluster = len(x) // 100

    # Run DBSCAN. The fit method will populate the .label_ parameter with 0 to n - 1, where n is the number of clusters.
    dbscan = DBSCAN(eps=0.025, min_samples=min_to_be_classified_as_a_cluster)
    dbscan.fit(matrix)

    clusters = {}
    for label_index in range(len(dbscan.labels_)):
        label = dbscan.labels_[label_index]

        # Add to `clusters` dict
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(matrix[label_index])

    return [np.array(cluster) for cluster in clusters.values() if len(cluster) >= 400]


def draw_bounding_box(data: np.ndarray, box_type: Callable) -> Union[OrientedBoundingBox, AxisAlignedBoundingBox]:
    bounding_box = box_type()

    pcd = o3d.utility.Vector3dVector(data)
    box = bounding_box.create_from_points(pcd)
    return box


def get_direction_angles(point: np.ndarray):
    magnitude = np.linalg.norm(point)
    return [np.degrees(np.arccos(i / magnitude)) for i in point]


def get_axes(points: np.ndarray, center: np.ndarray):
    p1, p2, p3, p4 = points[:4]
    sides = {np.linalg.norm(p2 - p1): (p2, p1), np.linalg.norm(p3 - p1): (p3, p1), np.linalg.norm(p4 - p1): (p4, p1)}

    side_order = list(sides.keys())
    side_order.sort()

    largest = side_order[2]
    next_largest = side_order[1]

    dir_vec = sides[largest][0] - sides[largest][1]
    side_vec = sides[next_largest][0] - sides[next_largest][1]

    print(get_direction_angles(side_vec))
    return lambda x: dir_vec * x + center, lambda y: side_vec * y + center


def main(filename):
    data_points = read_data(filename)

    ax = plt.axes(projection="3d")
    plt.title("Raw Data")
    ax.plot(data_points[:, 0], data_points[:, 1], data_points[:, 2], 'b.')
    plt.show()

    pcd_points = run_ransac(data_points, 0.02)
    ax = plt.axes(projection="3d")
    plt.title("Data after RANSAC")
    ax.plot(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2], 'b.')
    plt.show()

    filtered_clusters = run_dbscan(pcd_points)

    n = 1
    for points in filtered_clusters:
        bounding_box = draw_bounding_box(points, OrientedBoundingBox)
        box_points = np.array(bounding_box.get_box_points())
        box = [(box_points[i], box_points[j]) for i in range(box_points.shape[0]) for j in range(box_points.shape[0])]
        axis1, axis2 = get_axes(box_points, np.asarray(bounding_box.get_center()))

        print(len(points))

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # plt.title(f"Object {n}")

        ax.set_xlim3d(-0.2, 0)
        ax.set_ylim3d(-0.2, 0)
        ax.set_zlim3d(-0.5, -0.3)

        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b.')
        for pair in box:
            ax.plot((pair[0][0], pair[1][0]), (pair[0][1], pair[1][1]), (pair[0][2], pair[1][2]), 'r-')

        central_axis = np.array([axis1(i / 10) for i in range(-10, 10)])
        perp_axis = np.array([axis2(i / 10) for i in range(-10, 10)])
        ax.plot(central_axis[:, 0], central_axis[:, 1], central_axis[:, 2], 'mo')
        ax.plot(perp_axis[:, 0], perp_axis[:, 1], perp_axis[:, 2], 'go')
        plt.show()

        n += 1


if __name__ == "__main__":
    # for i in range(6):
    #     print(f"expected: {i + 1} items")
    #     main(f"Data/img{i}.ply")
    main("Data/img0.ply")
