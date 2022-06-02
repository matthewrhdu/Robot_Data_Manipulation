import numpy as np
import open3d as o3d


# filelist = [f"pos_chip_{i}.npy" for i in range(4 + 1)]
filelist = [f"img_pov{n}_items2.npy" for n in range(5)]
cali_list_xyz = [[+0.015, +0.005, -0.012]]


def merge_pointcloud(file_list):
    verts_merged = np.empty((0, 3), dtype=float)
    for idx, file in enumerate(file_list):
        verts_loaded = np.load(f"pov_images/{file}")
        for vid, vert in enumerate(verts_loaded):
            verts_loaded[vid] += cali_list_xyz[idx]
        verts_merged = np.append(verts_merged, verts_loaded, axis=0)

    return verts_merged


def main():
    verts = merge_pointcloud(filelist)
    verts_crop = np.empty((0, 3), dtype=float)

    for idx, vert in enumerate(verts):
        if not (0.6 > vert[0] > 0.2 and 0.3 > vert[1] > -1 and 0.3 > vert[2] > -0.08):
            verts[idx] = [0, 0, 0]
    print(verts.shape)
    verts = verts[~np.all(verts == 0, axis=1)]
    print(verts.shape)
    verts = np.unique(verts, axis=0)
    print(verts.shape)

    output_filename = "pos_chip_merged.npy"
    xfilename = "pos_chip_merged_x.npy"
    yfilename = "pos_chip_merged_y.npy"
    zfilename = "pos_chip_merged_z.npy"

    split_verts = np.hsplit(verts, 3)

    np.save(output_filename, verts)
    np.save(xfilename, split_verts[0])
    np.save(yfilename, split_verts[1])
    np.save(zfilename, split_verts[2])

    print("visualizing...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    o3d.io.write_point_cloud("../test.ply", pcd)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
