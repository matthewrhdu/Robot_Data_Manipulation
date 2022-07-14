import open3d as o3d
import numpy as np

# http://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html

bunny = o3d.data.BunnyMesh()
# mesh = o3d.io.read_triangle_mesh(bunny.path)
# mesh.compute_vertex_normals()
#
# pcd = mesh.sample_points_poisson_disk(750)
# o3d.visualization.draw_geometries([pcd], height=600, width=800)
# alpha = 0.03
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, height=600, width=800)
#
# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
#     print(f"alpha={alpha:.3f}")
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         pcd, alpha, tetra_mesh, pt_map)
#     mesh.compute_vertex_normals()
#     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, height=600, width=800)
#
# bunny = o3d.data.BunnyMesh()
# gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
# gt_mesh.compute_vertex_normals()
#
# pcd = gt_mesh.sample_points_poisson_disk(3000)
# o3d.visualization.draw_geometries([pcd], height=600, width=800)
#
# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([pcd, rec_mesh], height=600, width=800)

eagle = o3d.data.EaglePointCloud()
pcd = o3d.io.read_point_cloud(eagle.path)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001,
                                                      max_nn=300))


# pcd2 = o3d.io.read_point_cloud(eagle.path)
# pcd = o3d.io.read_point_cloud(bunny.path)

print(pcd)
o3d.visualization.draw_geometries([pcd],
                                  # zoom=0.664,
                                  # front=[-0.4761, -0.4698, -0.7434],
                                  # lookat=[1.8900, 3.2596, 0.9284],
                                  # up=[0.2304, -0.8825, 0.4101],
                                  height=600, width=800)

print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=2)
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  # zoom=0.664,
                                  # front=[-0.4761, -0.4698, -0.7434],
                                  # lookat=[1.8900, 3.2596, 0.9284],
                                  # up=[0.2304, -0.8825, 0.4101],
                                  height=600, width=800)
