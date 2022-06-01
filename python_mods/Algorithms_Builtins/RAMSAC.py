import open3d as o3d
import numpy as np


def run_ransac(pcd: np.ndarray, threshold: float, ransac_n: int = 3, num_iterations: int = 100):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)

    plane_model, inliers = point_cloud.segment_plane(distance_threshold=threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)

    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    return np.array(outlier_cloud.points)