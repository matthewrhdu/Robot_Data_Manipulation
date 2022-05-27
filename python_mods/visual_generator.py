import open3d as o3d


pcd = o3d.io.read_point_cloud(f"Data/img1.ply")
# o3d.visualization.draw_geometries([pcd])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=100)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])

outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 0, 1.0])

o3d.visualization.draw_geometries([outlier_cloud])

