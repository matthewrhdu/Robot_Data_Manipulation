import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

for i in range(5):
    data = np.load(f"object_{i}.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    mesh_obj = o3d.geometry.TriangleMesh()
    mesh = mesh_obj.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    mesh = mesh.compute_vertex_normals()

    pcd2 = mesh.sample_points_uniformly(10_000)

    hull = ConvexHull(pcd2.points)
    pcd2.points = o3d.utility.Vector3dVector(hull.points)
    # o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd2])
