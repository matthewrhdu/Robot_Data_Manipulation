import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def run_ransac(pcd: np.ndarray, threshold: float, ransac_n: int = 3, num_iterations: int = 100) -> tuple:
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

    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    return np.array(outlier_cloud.points), np.array(inlier_cloud.points)


def main(d_type=False):
    data = np.load(f"images2/img_combined.npy")
    ins, outs = run_ransac(data, 0.02)

    if d_type:
        ax = plt.axes(projection='3d')
        ax.plot(outs[:, 0], outs[:, 1], outs[:, 2], 'c.')
        ax.plot(ins[:, 0], ins[:, 1], ins[:, 2], 'b.')

        plt.xlabel("x")
        plt.ylabel("y")
        # ax.grid(False)
        # ax.set_axis_off()
        plt.show()
    else:
        in_cloud = o3d.geometry.PointCloud()
        in_cloud.points = o3d.utility.Vector3dVector(ins)

        out_cloud = o3d.geometry.PointCloud()
        out_cloud.points = o3d.utility.Vector3dVector(outs)
        out_cloud.paint_uniform_color([0.2, 0.8, 1])
        in_cloud.paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([in_cloud, out_cloud])


if __name__ == "__main__":
    main()

