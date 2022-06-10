import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from typing import Tuple


def setup_camera() -> Tuple[rs.pipeline, rs.config, rs.sensor]:
    """ Set up the camera for collecting point cloud information

    :return: The pipeline, configuration, and sensor
    """
    # Configure depth and color streams
    pipeline = rs.pipeline(None)
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for device_sensor in device.sensors:
        if device_sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break

    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    sensor = pipeline_profile.get_device().query_sensors()[0]

    sensor.set_option(rs.option.laser_power, 100)
    sensor.set_option(rs.option.confidence_threshold, 3)
    sensor.set_option(rs.option.min_distance, 150)
    sensor.set_option(rs.option.enable_max_usable_range, 0)
    sensor.set_option(rs.option.receiver_gain, 18)
    sensor.set_option(rs.option.post_processing_sharpening, 3)
    sensor.set_option(rs.option.pre_processing_sharpening, 5)
    sensor.set_option(rs.option.noise_filtering, 6)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    return pipeline, config, sensor


def capture_point_cloud_xyz(pipeline: rs.pipeline, config: rs.config) -> np.ndarray:
    """ Capture the xyz point cloud

    :param pipeline: The pipeline of the camera that is taking in information
    :param config: The configurations of the camera
    :return: The vertices of the point cloud
    """
    # Start streaming
    pipeline.start(config)

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()

    # Get the individual frames to be read
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_frame = decimate.process(depth_frame)

    color_image = np.asanyarray(color_frame.get_data())
    mapped_frame, color_source = color_frame, color_image

    points = pc.calculate(depth_frame)
    pc.map_to(mapped_frame)

    pipeline.stop()

    # Point cloud data to arrays
    vertices, texture_coordinates = points.get_vertices(), points.get_texture_coordinates()
    points = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)  # xyz

    return points


def main(num: int, visualize: bool = True) -> None:
    """ Starts the camera to process point clouds

    :param num: The point of view currently being observed. Position 1 is denoted `num = 0`
    :param visualize: Visualize the points if True.
    :return: None
    """
    pipeline, configs, sensor = setup_camera()
    transform_matrix = np.loadtxt(f"../TransformationMatrices/camera2base_matrix_for_image_{num}.txt")

    pcd_points = capture_point_cloud_xyz(pipeline, configs)

    arm_view_coordinates = []
    for pt in pcd_points:
        new_pt = np.matmul(transform_matrix, pt)
        arm_view_coordinates.append(new_pt)

    output_filename = f"image.npy"

    np.save(output_filename, np.array(arm_view_coordinates))
    print("visualizing...")

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arm_view_coordinates)
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main(0)
