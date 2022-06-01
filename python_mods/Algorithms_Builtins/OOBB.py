import open3d as o3d
import numpy as np
from typing import Callable, Union
from open3d.cpu.pybind.geometry import OrientedBoundingBox, AxisAlignedBoundingBox


def draw_bounding_box(data: np.ndarray, box_type: Callable) -> Union[OrientedBoundingBox, AxisAlignedBoundingBox]:
    bounding_box = box_type()

    pcd = o3d.utility.Vector3dVector(data)
    box = bounding_box.create_from_points(pcd)
    return box


def get_axis_lines(points: np.ndarray, center: np.ndarray):
    p1, p2, p3, p4 = points[:4]
    return lambda x: (p2 - p1) * x + center, lambda y: (p3 - p1) * y + center, lambda z: (p4 - p1) * z + center
