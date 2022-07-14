from in_hand_control.Cameras import PointCloudCamera
from in_hand_control.object_detection import segment_point_cloud

# PointCloudCamera.main(0, "image.npy")
segment_point_cloud.segment_point_cloud("../src/data")