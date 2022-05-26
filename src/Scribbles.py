import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def ransac2(data: np.ndarray, corners: list):
    first = np.median(np.array(corners)[:, 2])
    data_points = data
    good = []
    for point in data_points:
        diff = point - first
        if diff[2] > 0.025:
            good.append(point)

    return np.array(good)


