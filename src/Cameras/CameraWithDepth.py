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


def capture_pointcloud_xyz(pipeline, config):
    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    pipeline.get_active_profile()

    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Grab new intrinsics (may be changed by decimation)
    rs.video_stream_profile(depth_frame.profile).get_intrinsics()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    pipeline.stop()

    # Pointcloud data to arrays
    return None


def main(transform_position: int):
    pipeline, configs = setup_camera()
    transform_matrix = np.loadtxt(f"../TransformationMatrices/camera2base_matrix_for_image_{transform_position}.txt")
    verts = capture_pointcloud_xyz(pipeline, configs)

    new_verts = np.ndarray(shape=(verts.shape[0], verts.shape[1] + 1))

    for idx, vt in enumerate(verts):
        new_verts[idx] = np.append(vt, [1])
        new_verts[idx] = np.matmul(transform_matrix, new_verts[idx])
        verts[idx] = [new_verts[idx][0], new_verts[idx][1], new_verts[idx][2]]

    # output_filename = f"image{img_num}.npy"
    #
    # np.save(output_filename, verts)


if __name__ == '__main__':
    main(0)

