import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans, OPTICS
import matplotlib.pyplot as plt
from main2 import run_ransac
from random import randint

from open3d.cpu.pybind.geometry import OrientedBoundingBox, AxisAlignedBoundingBox
from typing import Callable, Union


def clean_data(data: np.ndarray):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x_upper = x <= 1
    x_lower = x >= -0.5
    y_upper = y <= 1
    y_lower = y >= -0.5

    mask = np.array([all([x_upper[i], x_lower[i], y_upper[i], y_lower[i]]) for i in range(len(x))])
    return np.column_stack((x[mask], y[mask], z[mask]))


def run_dbscan(point_cloud: np.ndarray):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    if len(x) // 2000 > 0:
        size = len(x) // 2000
    else:
        size = 1
    matrix = np.column_stack((x[::size], y[::size], z[::size]))

    # matrix = np.column_stack((x, y, z))
    min_to_be_classified_as_a_cluster = len(x) // 100

    # Run DBSCAN. The fit method will populate the .label_ parameter with 0 to n - 1, where n is the number of clusters.
    dbscan = DBSCAN(eps=0.025, min_samples=min_to_be_classified_as_a_cluster)
    # dbscan = KMeans(n_clusters=6, n_init=6, max_iter=100)
    dbscan.fit(matrix)

    clusters = {}
    for label_index in range(len(dbscan.labels_)):
        label = dbscan.labels_[label_index]

        # Add to `clusters` dict
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(matrix[label_index])

    print(len(clusters))

    return [np.array(item) for item in clusters.values() if len(item) >= 800]


def run_kmeans(point_cloud: np.ndarray):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    matrix = np.column_stack((x, y, z))

    # matrix = np.column_stack((x, y, z))
    min_to_be_classified_as_a_cluster = len(x) // 100

    # Run DBSCAN. The fit method will populate the .label_ parameter with 0 to n - 1, where n is the number of clusters.
    # dbscan = DBSCAN(eps=0.025, min_samples=min_to_be_classified_as_a_cluster)
    dbscan = KMeans(n_clusters=8, n_init=10, max_iter=100)
    dbscan.fit(matrix)

    clusters = {}
    for label_index in range(len(dbscan.labels_)):
        label = dbscan.labels_[label_index]

        # Add to `clusters` dict
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(matrix[label_index])

    print(len(clusters))

    return [np.array(item) for item in clusters.values() if len(item) >= 1000]


def draw_bounding_box(data: np.ndarray, box_type: Callable) -> Union[OrientedBoundingBox, AxisAlignedBoundingBox]:
    bounding_box = box_type()

    pcd = o3d.utility.Vector3dVector(data)
    box = bounding_box.create_from_points(pcd)
    return box


def get_axis_lines(points: np.ndarray, center: np.ndarray):
    p1, p2, p3, p4 = points[:4]
    return lambda x: (p2 - p1) * x + center, lambda y: (p3 - p1) * y + center, lambda z: (p4 - p1) * z + center


def main():
    # points = [0, 1, 2, 4]
    points = [0, 1]

    combined = []
    combined_img = []
    for i in points:
        pcd = np.load(f"pov_images/img_pov{i}_filtered.npy")
        cln = clean_data(pcd)
        rcd = run_ransac(cln, 0.02)
        data_ = rcd
        print(len(data_))

        combined.append(data_)
        combined_img.extend(data_)

    print(len(combined_img))

    # colours = [[1, 0.7, 0.7], [0.7, 1, 0.7], [0.7, 0.7, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
    # q = 0
    # vectors = []
    # for cluster in combined:
    #     vec = o3d.geometry.PointCloud()
    #     vec.points = o3d.utility.Vector3dVector(cluster)
    #     vec.paint_uniform_color(colours[q])
    #     vectors.append(vec)
    #     q += 1
    # o3d.visualization.draw_geometries(vectors, width=1920 // 2, height=1080 // 2)

    combined_img = np.array(combined_img)
    filtered_clusters = run_kmeans(combined_img)
    lineup = []
    for item in filtered_clusters[1:-1]:
        lineup.extend(run_dbscan(item))

    colours = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]] + [[randint(0, 100) / 100, randint(0, 100) / 100, randint(0, 100) / 100] for _ in range(10)]
    q = 0
    vectors = []
    for cluster in lineup:
        print(q, len(cluster))
        vec = o3d.geometry.PointCloud()
        vec.points = o3d.utility.Vector3dVector(cluster)
        vec.paint_uniform_color(colours[q])
        vectors.append(vec)
        q += 1
    o3d.visualization.draw_geometries(vectors, width=1920 // 2, height=1080 // 2)

    n = 1
    dataset = []
    for points in lineup:
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
    main()
