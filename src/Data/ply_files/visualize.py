import open3d as o3d

for i in range(5):
    data = o3d.io.read_point_cloud(f"img{i}.ply")
    o3d.visualization.draw_geometries([data], width=800, height=600)
