import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
# from src.point_cloud_segmentation.Algorithms_Builtins import run_ransac


def run_ransac(pcd: np.ndarray, threshold: float, ransac_n: int = 3, num_iterations: int = 100):
    """ Runs the RANSAC algorithm on the point cloud (`pcd`)

    :param pcd: The point cloud
    :param threshold: The threshold for the minimum distance for the outliers for the RANSAC algorithm
    :param ransac_n:  Number of initial points to be considered inliers in each iteration.
    :param num_iterations: Number of iterations of the RANSAC algorithm
    :return: A matrix of the points after the RANSAC algorithm
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)

    _, inliers = point_cloud.segment_plane(distance_threshold=threshold, ransac_n=ransac_n,
                                           num_iterations=num_iterations)

    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    inlier_cloud = point_cloud.select_by_index(inliers)
    return np.array(outlier_cloud.points), np.array(inlier_cloud.points)


def main():
    for i in range(5):
        data = np.load(f"image{i}_cleaned.npy")
        outs, ins = run_ransac(data, 0.01)
        # outs = data

        out_cloud = o3d.geometry.PointCloud()
        out_cloud.points = o3d.utility.Vector3dVector(outs)
        out_cloud.paint_uniform_color([0, 0, 1])

        in_cloud = o3d.geometry.PointCloud()
        in_cloud.points = o3d.utility.Vector3dVector(ins)
        in_cloud.paint_uniform_color([0.3, 0.8, 1])

        o3d.visualization.draw_geometries([out_cloud, in_cloud])
    # np.save(f"image{i}_ransaced_cleaned.npy", outs)


if __name__ == "__main__":
    main()
