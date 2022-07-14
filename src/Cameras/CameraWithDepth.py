import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import open3d as o3d


def setup_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
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
    return pipeline, config


def capture_pointcloud_xyz(pipeline, config):
    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    pipeline.get_active_profile()

    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Grab new intrinsics (maybe changed by decimation)
    rs.video_stream_profile(depth_frame.profile).get_intrinsics()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    pipeline.stop()

    # Point cloud data to arrays
    return color_image, depth_image


def process_positions(img: np.ndarray):
    draw_img = np.copy(img)
    img_gray = cv.cvtColor(draw_img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
    img_gray = cv.Canny(img_gray, 100, 200)
    _, thresh = cv.threshold(img_gray, 127, 255, 0)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(draw_img, contours, -1, (0, 255, 0), 3)
    cv.imshow("image", draw_img)
    cv.waitKey(0)


def show_depths(img: np.ndarray):
    points = np.array([[x, y, img[y][x]] for y in img.shape[0] for x in img.shape[1]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def main(transform_position: int):
    pipeline, configs = setup_camera()
    # transform_matrix = np.loadtxt(f"../TransformationMatrices/camera2base_matrix_for_image_{transform_position}.txt")
    rgb_image, depth_image = capture_pointcloud_xyz(pipeline, configs)

    for row in depth_image: print(row)
    cv.imshow("img", rgb_image  )
    cv.waitKey(0)


if __name__ == '__main__':
    main(0)

