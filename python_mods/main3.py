import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from open3d.cpu.pybind.geometry import OrientedBoundingBox, AxisAlignedBoundingBox
from typing import Callable, Union


def read_data(filename: str) -> np.ndarray:
    """ Read the .ply file with filename (filename includes .ply) and returns an o3d point cloud"""
    if filename[-len(".ply"):] == ".ply":
        return np.asarray(o3d.io.read_point_cloud(filename).points)
    elif filename[-len(".npy"):] == ".npy":
        return np.load(filename)
    else:
        raise FileNotFoundError

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
    dbscan = DBSCAN(eps=0.05, min_samples=min_to_be_classified_as_a_cluster)
    dbscan.fit(matrix)

    clusters = {}
    for label_index in range(len(dbscan.labels_)):
        label = dbscan.labels_[label_index]

        # Add to `clusters` dict
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(matrix[label_index])

    print(len(clusters))

    return [np.array(item) for item in clusters.values() if len(item) >= 400]


def draw_bounding_box(data: np.ndarray, box_type: Callable) -> Union[OrientedBoundingBox, AxisAlignedBoundingBox]:
    bounding_box = box_type()

    pcd = o3d.utility.Vector3dVector(data)
    box = bounding_box.create_from_points(pcd)
    return box


def get_direction_angles(point: np.ndarray):
    magnitude = np.linalg.norm(point)
    return [np.degrees(np.arccos(i / magnitude)) for i in point]


def get_axis_lines(points: np.ndarray, center: np.ndarray):
    p1, p2, p3, p4 = points[:4]
    return lambda x: (p2 - p1) * x + center, lambda y: (p3 - p1) * y + center, lambda z: (p4 - p1) * z + center


def main2(filename):
    data_points = read_data(filename)
    print(data_points)
    # pcd_points = run_ransac(data_points, 0.025)

    pcd_points = data_points
    filtered_clusters = run_dbscan(pcd_points)

    n = 1
    dataset = []
    for points in filtered_clusters:
        bounding_box = draw_bounding_box(points, OrientedBoundingBox)
        box_pts = np.array(bounding_box.get_box_points())

        box_points = box_pts
        box_center = np.asarray(bounding_box.get_center())

        # box = [(box_points[i], box_points[j]) for i in range(box_points.shape[0]) for j in range(box_points.shape[0])]
        bbl = box_points[0]
        bbr = box_points[1]
        btl = box_points[2]
        fbl = box_points[3]
        ftr = box_points[4]
        ftl = box_points[5]
        fbr = box_points[6]
        btr = box_points[7]
        box = [(bbl, bbr), (btl, btr), (bbl, btl), (bbr, btr), (fbl, fbr), (ftl, ftr), (fbl, ftl), (fbr, ftr),
               (bbl, fbl), (btl, ftl), (btr, ftr), (bbr, fbr)]

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
    main2("combined.npy")
