import open3d as o3d
import copy
import numpy as np
from typing import List, Tuple


def _np_leq(arr1: np.ndarray, arr2: np.ndarray, pos: int = 0) -> bool:
    """ A helper function for sorting NumPy arrays. Returns True if arr1 <= arr2.

    A NumPy array is less than another by comparing element by element:
    - If the element at index `pos` of arr1 is less than the item at index `pos` of arr2, return true
    - If the element at index `pos` of arr1 is equal to the item at index `pos` of arr2, search the next element

    :param arr1: The first array
    :param arr2: The second array
    :param pos: The position to be searched. Used for recursion
    :return: True if arr1 <= arr2
    """
    if arr1[pos] <= arr2[pos]:
        return True
    elif arr1[pos] > arr2[pos]:
        return False
    else:
        if pos + 1 > 2:
            return True
        return _np_leq(arr1, arr2, pos + 1)


def _merge(lst1: List[np.ndarray], lst2: List[np.ndarray]) -> List[np.ndarray]:
    """ Return a sorted list with the elements in `lst1` and `lst2`.

    Precondition: `lst1` and `lst2` are sorted.

    :param lst1: The first list
    :param lst2: The second list
    :return: lst1 and lst2 merged
    """
    index1 = 0
    index2 = 0
    merged = []
    while index1 < len(lst1) and index2 < len(lst2):
        if _np_leq(lst1[index1], lst2[index2]):
            merged.append(lst1[index1])
            index1 += 1
        else:
            merged.append(lst2[index2])
            index2 += 1

    # now either index1 == len (lst1) or index2 == len (lst2).
    assert index1 == len(lst1) or index2 == len(lst2)
    return merged + lst1[index1:] + lst2[index2:]


def merge_sort(lst: List[np.ndarray]) -> List[np.ndarray]:
    """ Sort `lst` using the mergesort algorithm

    :param lst: The list to be sorted
    :return: The sorted List
    """
    if len(lst) < 2:
        return lst[:]
    else:
        m = len(lst) // 2
        left_lst = lst[:m]
        right_lst = lst[m:]

        left_sorted = merge_sort(left_lst)
        right_sorted = merge_sort(right_lst)

        return _merge(left_sorted, right_sorted)


def write_point_clouds(first: o3d.geometry.PointCloud, second: o3d.geometry.PointCloud) -> None:
    """ Write the point clouds first and second appended together to disk

    :param first: The first point cloud
    :param second: The second point cloud
    :return: None
    """
    pcd1, pcd2 = np.asarray(first.points[::2]), np.asarray(second.points[::2])
    arr = pcd1.tolist() + pcd2.tolist()

    sorted_write_arr = merge_sort(arr)
    write_arr = []
    curr = sorted_write_arr[0]
    for item in sorted_write_arr[1:]:
        if not np.array_equal(curr, item):
            write_arr.append(item)
        curr = item

    write_arr = np.asarray(write_arr)
    print(f"writing: {len(write_arr)}")
    np.save("../Data/pov_images/images2/img_combined.npy", write_arr)


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                             transformation: np.ndarray) -> None:
    """ Draw the point clouds for visualization purposes

    :param source: The point cloud of the source
    :param target: The point cloud of the target
    :param transformation: The transformation matrix to apply
    :return: None
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp], width=1920 // 2, height=1080 // 2)


def draw_and_save_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                                      transformation: np.ndarray) -> None:
    """ Draw and save the point clouds for visualization purposes

    :param source: The point cloud of the source
    :param target: The point cloud of the target
    :param transformation: The transformation matrix to apply
    :return: None
    """
    draw_registration_result(source, target, transformation)
    write_point_clouds(source, target)


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> Tuple[o3d.geometry.PointCloud,
                                                                                     o3d.pipelines.registration.Feature]:
    """ Preprocess the point cloud and generate a Fast Point Feature Histogram (fpfh)

    :param pcd: The point cloud to preprocess
    :param voxel_size: The size of the voxels to process
    :return: The down-sampled point cloud, the fpfh of the point cloud
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 4
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 10
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size: float, filename1: str, filename2: str):
    """ Prepare a dataset for point cloud registration

    :param voxel_size: The size of the voxel
    :param filename1: The filename (with .npy ending) of dataset 1
    :param filename2: The filename (with .npy ending) of dataset 2
    :return: ...
    """
    first = np.load(filename1)
    second = np.load(filename2)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(first)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(second)

    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.2

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 5
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result


def main(filename_1: str, filename_2: str, voxel_size: float = 0.005):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, filename_1, filename_2)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)
    print(result_icp)
    draw_and_save_registration_result(source, target, result_icp.transformation)


if __name__ == "__main__":
    pass
    # main("../Data/pov_images/images2/image0_ransaced_cleaned.npy", "../Data/pov_images/images2/image1_ransaced_cleaned.npy")
    # for i in range(2, 5):
    #     main("../Data/pov_images/images2/img_combined.npy", f"../Data/pov_images/images2/image{i}_ransaced_cleaned.npy")

