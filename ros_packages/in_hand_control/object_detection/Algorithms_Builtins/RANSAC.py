import open3d as o3d
import numpy as np


def run_ransac(pcd: np.ndarray, threshold: float, ransac_n: int = 3, num_iterations: int = 100) -> np.ndarray:
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
    return np.array(outlier_cloud.points)
