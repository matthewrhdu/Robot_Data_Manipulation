import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.spatial import ConvexHull

from typing import Tuple


debug = True


def _get_rotation_matrix(theta: float, x: float, y: float) -> np.ndarray:
    """
    The matrix is;
    [[ cos t    - sin t     x cos t - y sin t]
     [ sin t    cos t       x sin t + y cos t]
     [ 0        0           0               ]]

    :param theta: angle (in **DEGREES**)
    :param x: rotate around point x coordinate
    :param y: rotate around point y coordinate
    :return: Matrix
    """
    sin_theta = np.sin(np.radians(theta))
    cos_theta = np.cos(np.radians(theta))
    return np.array([
        [cos_theta, - sin_theta, x * cos_theta - y * sin_theta],
        [sin_theta, cos_theta, x * sin_theta + y * cos_theta],
        [0, 0, 0]
    ])


def _multiply_point_set(matrix: np.ndarray, pt_set: np.ndarray) -> np.ndarray:
    """ Multiply all points in pt_set by matrix. If the sizes of the points do not match with the matrix, append a 1
    at the end for each addition new space needed

    :param matrix: The matrix to multiply each point by
    :param pt_set: The set of points
    :return: The new set of points post multiplication
    """
    new_pt_set = []
    for pt in pt_set:
        # If the size of the point does not match that of the matrix
        if pt.shape[0] != matrix.shape[0]:
            new_pt = np.array([pt[0], pt[1], 1])
        else:
            new_pt = pt

        new_pt_set.append(np.matmul(matrix, new_pt))

    return np.array(new_pt_set)


def _get_centroid(pcd: np.ndarray) -> Tuple[ConvexHull, float, float]:
    """ Gets the centroid of a point cloud `pcd`

    :param pcd: Point cloud to obtain centroid from
    :return: The Convexhull to generate the centroid; the centroid x and y components
    """
    hull = ConvexHull(pcd)

    cx = np.mean(hull.points[hull.vertices, 0])
    cy = np.mean(hull.points[hull.vertices, 1])

    return hull, float(cx), float(cy)


def _count_inliers(pcd: np.ndarray, target_hull: ConvexHull) -> int:
    """ Count the number of points from post_rotation_points within the range of target_hull

    :param pcd: The point cloud of points in the source
    :param target_hull: The Convex hull of the target
    :return: Number of points from post_rotation_points in target_hull
    """
    acc = 0
    for point in pcd:
        in_range = target_hull.min_bound[0] <= point[0] <= target_hull.max_bound[0] and \
                   target_hull.min_bound[1] <= point[1] <= target_hull.max_bound[1]
        acc += sum([1 if in_range else 0])
    return acc


def find_contours(img: np.ndarray) -> np.ndarray:
    """ Find the contours from a given image `img`

    :param img: The observed image
    :return: The array of all contours in the image
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 5)
    img_gray = cv.Canny(img_gray, 100, 200)
    _, thresh = cv.threshold(img_gray, 127, 255, 0)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_objects(contours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the target and source objects from the `contours` array

    :param contours: The array of points for a contour
    :return: a tuple of the source object and the target object **in that order**
    """
    first = contours[0][:, 0]
    second = None
    first_center = np.array([int(np.average(first[0])), int(np.average(first[1]))])
    for contour in contours:
        center = np.array([int(np.average(contour[:, 0][0])), int(np.average(contour[:, 0][1]))])

        # Ensures all center is at least 10 pixels away from each other (|pt - center| > 10)
        if abs(np.linalg.norm(center - first_center)) > 10:
            second = contour[:, 0]
            break

    # Safety check
    assert second is not None

    return first, second


def get_best_fit(target_center_x: float, target_center_y: float, target_hull: ConvexHull, pcd: np.ndarray) \
        -> Tuple[int, np.ndarray]:
    """ Gets the best fit of after rotation for pcd

    :param target_center_x: The x position of the centroid of the target
    :param target_center_y: The y position of the centroid of the target
    :param target_hull: The convex hull encompassing the target
    :param pcd: The point cloud of points of the source
    :return: The angle of best fit, followed by the points after rotation
    """
    angle_fitness = {}
    rotated_pcds = []
    for theta in range(360):
        rotation_matrix = _get_rotation_matrix(theta, target_center_x, target_center_y)

        post_rotation_points = _multiply_point_set(rotation_matrix, pcd)
        rotated_pcds.append(post_rotation_points)

        num_inliers = _count_inliers(post_rotation_points, target_hull)

        if num_inliers not in angle_fitness:
            angle_fitness[num_inliers] = []
        angle_fitness[num_inliers].append(theta)

    best_fit = angle_fitness[max(angle_fitness.keys())]
    best_fit_pcd = rotated_pcds[best_fit[0]]

    return best_fit[0], best_fit_pcd


def main() -> int:
    """ Runs the program

    :return: Exit code
    """
    img = cv.imread("img.png")
    contours = find_contours(img)

    source, target = get_objects(contours)

    hull1, cx1, cy1 = _get_centroid(source)
    hull2, cx2, cy2 = _get_centroid(target)

    transformation_matrix = np.array([
        [1, 0, cx2 - cx1],
        [0, 1, cy2 - cy1],
        [0, 0, 0]
    ])

    new_first = _multiply_point_set(transformation_matrix, source)

    if debug:
        plt.plot(target[:, 0], target[:, 1], 'r.')
        plt.plot(source[:, 0], source[:, 1], 'g.')
        plt.plot(new_first[:, 0], new_first[:, 1], 'bo')
        plt.show()

    best_fit, new_first = get_best_fit(cx2, cy2, hull2, new_first)

    print(np.matmul(_get_rotation_matrix(best_fit, cx2, cy2), transformation_matrix))

    plt.plot(target[:, 0], target[:, 1], 'r.')
    plt.plot(new_first[:, 0], new_first[:, 1], 'bo')
    plt.show()

    return 0


if __name__ == "__main__":
    main()
