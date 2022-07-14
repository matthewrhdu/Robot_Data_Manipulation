import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import OrientedBoundingBox, AxisAlignedBoundingBox
from typing import Union, Tuple, List


def draw_bounding_box(data: np.ndarray, box_type: Union[OrientedBoundingBox, AxisAlignedBoundingBox]) -> \
        Union[OrientedBoundingBox, AxisAlignedBoundingBox]:
    """ Draw a bounding box of type `box_type` around `data`

    :param data: The matrix of the points of the object that the bounding box will cover
    :param box_type: The type of box that will be used. Must be `OrientatedBoundingBox` or `AxisAlignedBoundingBox`

    :return: a bounding box object of a bounding box type
    """
    bounding_box = box_type
    pcd = o3d.utility.Vector3dVector(data)
    box = bounding_box.create_from_points(pcd)
    return box


def get_basis_vectors(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get the basis vectors for the subspace of the `points` of the point clouds

    :param points: The points of the point cloud
    :return: The three basis vectors of the point cloud in no particular order
    """
    p1, p2, p3, p4 = points[:4]  # Defensive programming
    return p2 - p1, p3 - p1, p4 - p1


def get_box_pairs(box_points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ Gets the pair of points that define the edges of the bounding box for visualization purposes

    :param box_points: The points of the bounding box, from the output of the Bounding Box Algorithm
    :return: A list of corner pairs that define the sides of the bounding box
    """
    bbl = box_points[0]
    bbr = box_points[1]
    btl = box_points[2]
    fbl = box_points[3]
    ftr = box_points[4]
    ftl = box_points[5]
    fbr = box_points[6]
    btr = box_points[7]
    return [(bbl, bbr), (btl, btr), (bbl, btl), (bbr, btr), (fbl, fbr), (ftl, ftr), (fbl, ftl), (fbr, ftr), (bbl, fbl),
            (btl, ftl), (btr, ftr), (bbr, fbr)]

