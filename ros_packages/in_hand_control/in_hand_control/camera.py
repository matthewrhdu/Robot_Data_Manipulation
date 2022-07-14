import rclpy
from in_hand_control.device import Device
# from services.srv import DoStuff
from services.action import RunCamera
import cv2 as cv
import pyrealsense2 as rs
import cv2
import numpy as np

def capture_images(pipeline):
    ## Get frameset of color
    frames = pipeline.wait_for_frames()  
    color_frame = frames.get_color_frame()   # get color frame

    ## Get the depth and color image and Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())  # color image
    image_resized = cv2.resize(color_image, (int(1280 / 2), int(720 / 2)))

    return color_image, image_resized

def stuff():
    ## Configure depth and color streams
    pipeline = rs.pipeline()  #Create a realsenes pipeline
            
    config = rs.config()   # Create a config
            
    ## Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    #for our camera L515, the device_product_line == 'L500'
            
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        raise Exception('No RGB camera found!')
        # exit(0)
            
    ## define the sensor parameter for close range image process
    sensor = pipeline_profile.get_device().query_sensors()[0]

    sensor.set_option(rs.option.laser_power, 100)
    sensor.set_option(rs.option.confidence_threshold, 1)
    sensor.set_option(rs.option.min_distance, 0)
    sensor.set_option(rs.option.enable_max_usable_range, 0)
    sensor.set_option(rs.option.receiver_gain, 18)
    sensor.set_option(rs.option.post_processing_sharpening, 3)
    sensor.set_option(rs.option.pre_processing_sharpening, 5)
    sensor.set_option(rs.option.noise_filtering, 6)

    ## Configure the pipeline to stream different resolutions of color and depth streams
            
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            
    ## Start streaming
    profile = pipeline.start(config)  # start the pipeline
    
    color_image, image_resized = capture_images(pipeline)
    
    cv2.imwrite("image.png", color_image)
    

class Camera(Device):
    def __init__(self):
        super().__init__('camera', 'camera_command', RunCamera)

    def run(self, msg):
        self.get_logger().info("Received Request")

        result = RunCamera.Result()
        if msg.request.configuration == 0:
            self.get_logger().info("Click")
            # stuff()
            result.exit_status = 0
        else:
            result.exit_status = 1
        
        msg.succeed()

        return result

    def take_picture(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().info("Cannot open camera")
            exit(1)

        # Capture frame-by-frame
        ret, img = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            self.get_logger().info("Can't receive frame (stream end?). Exiting ...")
    
        cv.imwrite("image.png", img)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)

    camera = Camera()
    try:
        rclpy.spin(camera)
    except KeyboardInterrupt:
        pass

    camera.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()