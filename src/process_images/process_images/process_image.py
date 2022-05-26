import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from image_data.srv import Position


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


def draw_bounding_box(data: np.ndarray, box_type):
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


class ProcessImage(Node):
    def __init__(self):
        super().__init__('process_image')
        self.cli = self.create_client(Position, 'position')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Position.Request()

    def send_request(self):
        self.req.x_pos = 0.0
        self.req.y_pos = 0.0
        self.req.z_pos = 0.0
        self.future = self.cli.call_async(self.req)


def do_useful_stuff(process_image):
    data_points = np.load("data.npy")
    pcd_points = run_ransac(data_points, 0.02)
    filtered_clusters = run_dbscan(pcd_points)

    for points in filtered_clusters:
        bounding_box = draw_bounding_box(points, o3d.geometry.OrientedBoundingBox)
        box_points = np.array(bounding_box.get_box_points())
        box_center = np.array(bounding_box.get_center())

        angles = get_axes(box_points)
        process_image.get_logger().info(str(box_center.tolist()))
        process_image.get_logger().info(str(angles))


def main(args=None):
    rclpy.init(args=args)

    process_image = ProcessImage()
    process_image.send_request()

    while rclpy.ok():
        rclpy.spin_once(process_image)
        if process_image.future.done():
            exit_status = process_image.future.result().exit_status
            process_image.get_logger().info(f"Exit Status {exit_status} \n")
            if exit_status == 0:
                do_useful_stuff(process_image)
            else:
                process_image.get_logger().info("something happened. Try again!")
                exit(1)
            break

    process_image.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()