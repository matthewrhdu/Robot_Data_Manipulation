import open3d as o3d
import numpy as np

# http://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html

data = np.load("object_2_cleaned.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
pcd.paint_uniform_color([1, 0, 0])

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001,
                                                      max_nn=30))

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])
