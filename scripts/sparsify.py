import open3d as o3d
import numpy as np


if __name__ == "__main__":
    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud("capoo.ply")
    o3d.io.write_point_cloud("capoo_sparse.ply", pcd.voxel_down_sample(voxel_size=0.2))
