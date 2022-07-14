import numpy as np
import pyrealsense2 as rs


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


def start_pipeline(pipeline, config):
    pipeline.start(config)


def stop_pipeline(pipeline):
    pipeline.stop()

def get_point_cloud(transform_position: int, pipeline):
    transform_matrix = np.loadtxt(f"../TransformationMatrices/camera2base_matrix_for_image_{transform_position}.txt")

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()

    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_frame = decimate.process(depth_frame)

    color_image = np.asanyarray(color_frame.get_data())

    mapped_frame, color_source = color_frame, color_image

    points = pc.calculate(depth_frame)
    pc.map_to(mapped_frame)

    # Point cloud data to arrays
    v, t = points.get_vertices(), points.get_texture_coordinates()
    vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

    transformed_points = np.ndarray(shape=(vertices.shape[0], vertices.shape[1] + 1))

    for pt_index, points in enumerate(vertices):
        temp = np.append(points, [1])
        transformed_points[pt_index] = np.matmul(transform_matrix, temp)

    return transformed_points


if __name__ == '__main__':
    pass

