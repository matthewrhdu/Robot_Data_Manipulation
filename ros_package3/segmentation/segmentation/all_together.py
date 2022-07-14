import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription, MsgType

from std_msgs.msg import Int64

import numpy as np
from typing import Optional
import open3d as o3d
from threading import Thread
from queue import Queue
from typing import List
from pyrealsense2 import pipeline

from RANSAC import run_ransac
import global_registration2 as icp_registration
import PointOfViewCamera as Camera


NUM_VIEWS = 5
DIMENSIONALITY = 3


def _filter(pcd: np.ndarray, filter_spec: List[List[float]]) -> np.ndarray:
    """ Filter the point cloud (pcd) by removing all points outside the specs defined in filter_specs

    >>> arr = np.ndarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    >>> cleaned = _filter(arr, [[0., 0.5], [0., 1.], [0., 1.]])
    [[0. 1. 0.]
     [0. 0. 1.]]

    :param pcd: The point cloud to be filtered
    :param filter_spec: The specification list of the range of data to be filtered. The list should be in the form:
        [[min(x), max(x)], [min(y), max(y)], [min(z), max(z)]]```
    :return: The filtered point cloud
    """
    # Create a mask with regex 1*
    mask = np.full((pcd.shape[0],), True)

    for i in range(DIMENSIONALITY):
        lower_mask = pcd[:, i] >= filter_spec[i][0]
        upper_mask = pcd[:, i] <= filter_spec[i][1]
        range_mask = np.logical_and(lower_mask, upper_mask)

        mask = np.logical_and(mask, range_mask)

    return np.array([pcd[m] for m in range(len(mask)) if mask[m]])


def _get_statistical_outliers(pcd: np.ndarray) -> List[List[float]]:
    """ Get the statistical outliers from the point cloud

    :param pcd: The point cloud
    :return: A filter specification list of all the outliers in the array
    """
    stat_outlier_arr = []
    for i in range(DIMENSIONALITY):
        mean = np.average(pcd[:, i])
        std = np.std(pcd[:, i])

        stat_outlier_arr.append([mean - std, mean + std])

    return stat_outlier_arr


class MergedImage(Node):
    """ A ROS2 Node to merge images

    === Attributes ===
    :ivar accumulated: The accumulator. Keeps track of the current scene and is the image that all other images are
        merged against. Is None if it is the first image
    :ivar subscription: A ROS subscription to receive the start signal by the arm to start taking pictures
    :ivar lineup: A Queue of `NUM_VIEWS` items to keep track of the point clouds that has been read by the camera that
        is yet to be processed
    :ivar boundaries: The list of boundary conditions derived from the first image taken. The boundaries are used as
        the domain and range of the xy plane for all subsequent images taken
    :ivar images_taken: The variable accumulator to keep track of the number of views already taken.
    :ivar images_merged: The number of images that has already been merged
    :ivar camera_pipeline: The pipeline from the pyrealsense camera that takes pictures
    """
    accumulated: Optional[np.ndarray]
    subscription: Subscription
    lineup: Queue
    boundaries: List[List[float]]
    images_taken: int
    images_merged: int
    camera_pipeline: pipeline

    def __init__(self) -> None:
        """ Initializer """
        super().__init__("merge_image")
        self.accumulated = None
        self.subscription = self.create_subscription(Int64, 'a', self.take_picture_callback, 10)
        self.lineup = Queue(NUM_VIEWS)
        self.boundaries = [[], [], []]
        self.images_taken = 0
        self.images_merged = 0
        self.camera_pipeline, configs = Camera.setup_camera()
        Camera.start_pipeline(self._pipeline, configs)

    def take_picture_callback(self, msg: MsgType) -> None:
        """ The callback function for the subscriber. Takes a picture for the camera

        :param msg: The message received by the camera. At the current state, the data in the message does not matter.
        :return Nothing
        """
        self.get_logger().info(f"I received message {msg.data}")
        self.lineup.put(Camera.get_point_cloud(self.images_taken, self._pipeline))
        self.get_logger().info("Click!!!")

    def accumulate(self) -> None:
        """ The accumulation function. Accumulate and processes images in the node.

        Saves the images from each step in the same directory

        :return: Nothing
        """
        # Wait until the first image is taken
        img = self.lineup.get(block=True)

        plane_removed_img = run_ransac(img, 0.0075)

        # If this is the first image. Second condition is a failsafe / defensive programming and is not needed
        if self.accumulated is None and self.images_taken == 0:
            self.accumulated = plane_removed_img
            self._get_boundaries(plane_removed_img)

        # All the other images
        else:
            # Gets the points in bound
            filtered = _filter(plane_removed_img, self.boundaries)

            filter_stat = _get_statistical_outliers(filtered)
            filtered = _filter(filtered, filter_stat)

            self.accumulated = icp_registration.main(filtered, self.accumulated, voxel_size=0.005)

            np.save(f"sub/sub{self.images_merged}.npy", self.accumulated)
            self.images_merged += 1

    def _get_boundaries(self, pcd: np.ndarray) -> None:
        """ Get the domain of the point cloud pcd. Sets the domain in the boundaries attribute

        :param pcd: The point cloud
        :return: Nothing
        """
        for i in range(DIMENSIONALITY):
            self.boundaries[i] = [min(pcd[:, i]), max(pcd[:, i])]


def camera_thread(machine: MergedImage) -> None:
    """ The thread for the camera node. Photos are taken in this thread.

    :param machine: The ros node that takes the images
    :return: Nothing. Everything is done in the MergedImage object
    """
    for n in range(NUM_VIEWS):
        rclpy.spin_once(machine)
    Camera.stop_pipeline(machine.camera_pipeline)


def main(args=None) -> int:
    """ The main body of execution for the program

    :param args: The arguments that needs to be initialized for the rclpy.
    :return: Exit status
    """
    rclpy.init(args=args)
    machine = MergedImage()

    # Starts parallel thread
    parallel_thread = Thread(target=camera_thread, args=(camera_thread,))
    parallel_thread.start()

    for _ in range(NUM_VIEWS):
        machine.accumulate()

    # Closes the parallel thread
    parallel_thread.join()

    # Visualizes the point cloud after completion
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(machine.accumulated)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)

    return 0


if __name__ == "__main__":
    main()
