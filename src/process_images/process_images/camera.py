import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d


from image_data.srv import Position

def read_data(filename: str) -> np.ndarray:
    """ Read the .ply file with filename (filename includes .ply) and returns an o3d point cloud"""
    return np.asarray(o3d.io.read_point_cloud(filename).points)

class Camera(Node):
    def __init__(self):
        super().__init__('Camera')
        self.srv = self.create_service(Position, 'position', self.camera_callback)

    def camera_callback(self, request, response):
        self.get_logger().info("Message Received. Creating File")
        position = [request.x_pos, request.y_pos, request.z_pos]
        data = read_data("Data/img0.ply")
        if position == [0.0, 0.0, 0.0]:
            np.save("data.npy", data)
            response.exit_status = 0
        else:
            response.exit_status = 1

        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = Camera()
    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()