import numpy as np
import open3d as o3d
from _3D_Object_Recognition.Algorithms_Builtins.RAMSAC import run_ransac


def clean_data(data: np.ndarray):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x_upper = x <= 1
    x_lower = x >= 0.15
    y_upper = y <= 1
    y_lower = y >= -0.5

    mask = np.array([all([x_upper[i], x_lower[i], y_upper[i], y_lower[i]]) for i in range(len(x))])
    return np.column_stack((x[mask], y[mask], z[mask]))


def clean_all():
    for i in range(5):
        info = np.load(f"img_pov{i}_items2.npy")
        info = clean_data(info)
        info = run_ransac(info, 0.025)
        np.save(f"img_pov{i}_filtered.npy", info)


def clean2(pcd_: np.ndarray):
    x, y, z = pcd_[:, 0], pcd_[:, 1], pcd_[:, 2]

    mask = z >= min(z) + 0.25
    return np.column_stack((x[mask], y[mask], z[mask]))


if __name__ == "__main__":
    img_num = 3
    data_pcd = o3d.io.read_point_cloud(f"sample{img_num}.ply")
    data = np.asarray(data_pcd.points)

    i = 0
    mat = np.loadtxt(f"../TransformationMatrices/camera2base_matrix_for_image_{i}.txt")

    data = clean_data(data)
    # data = clean2(data)
    data = run_ransac(data, 0.01)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)

    np.save(f"img_pov{img_num}_filtered.npy", data)

