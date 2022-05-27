import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
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

    return np.array([np.array(cluster) for cluster in clusters.values() if len(cluster) >= 400])


def draw_bounding_box(data: np.ndarray, box_type: Callable) -> Union[OrientedBoundingBox, AxisAlignedBoundingBox]:
    bounding_box = box_type()

    pcd = o3d.utility.Vector3dVector(data)
    box = bounding_box.create_from_points(pcd)
    return box


def get_direction_angles(point: np.ndarray):
    magnitude = np.linalg.norm(point)
    return [np.degrees(np.arccos(i / magnitude)) for i in point]


def get_axes(points: np.ndarray):
    p1, p2, p3, p4 = points[:4]
    sides = {np.linalg.norm(p2 - p1): (p2, p1), np.linalg.norm(p3 - p1): (p3, p1), np.linalg.norm(p4 - p1): (p4, p1)}

    side_order = list(sides.keys())
    side_order.sort()

    next_largest = side_order[1]

    side_vec = sides[next_largest][0] - sides[next_largest][1]

    return get_direction_angles(side_vec)


def main(filename):
    data_points = read_data(filename)
    pcd_points = run_ransac(data_points, 0.02)
    filtered_clusters = run_dbscan(pcd_points)

    for points in filtered_clusters:
        bounding_box = draw_bounding_box(points, OrientedBoundingBox)
        box_points = np.asarray(bounding_box.get_box_points())
        box_center = np.array(bounding_box.get_center())

        angles = get_axes(box_points)

        transformation_matrix = np.array([
            np.array([-1, 0, 0, box_center[0]]),
            np.array([0, -1, 0, box_center[1]]),
            np.array([0, 0, -1, box_center[2]])
        ])

        print(transformation_matrix)
        print(angles)


if __name__ == "__main__":
    # for i in range(6):
    #     print(f"expected: {i + 1} items")
    #     main(f"Data/img{i}.ply")
    main("Data/img0.ply")
