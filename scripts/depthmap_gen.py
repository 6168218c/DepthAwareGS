import numpy as np
import cv2
from captured_data import read_calib_data
import open3d as o3d


def gen_depthmap(plyfile, calibfile, outputfile):
    pcd = o3d.io.read_point_cloud(plyfile)

    calib_data = read_calib_data(calibfile)
    K, R, T, width, height = (
        calib_data["cam_mtx"],
        calib_data["R"],
        calib_data["T"],
        int(calib_data["width"]),
        int(calib_data["height"]),
    )
    resolution = 1

    # K[:2, 2] = [width // 2, height // 2]

    camera2projector = np.zeros((4, 4), dtype=np.float32)
    camera2projector[:3, :3] = R
    camera2projector[:3, 3] = T.squeeze()
    camera2projector[3, 3] = 1
    projector2camera = np.linalg.inv(camera2projector)

    R, T = projector2camera[:3, :3], projector2camera[:3, 3:]

    depthmap, depth_weight = (
        np.zeros((height // resolution, width // resolution)),
        np.zeros((height // resolution, width // resolution)),
    )
    cam_coord = np.matmul(R, np.asarray(pcd.points).transpose()) + T.reshape(3, 1)
    projected_coord = np.matmul(K, cam_coord)
    valid_idx = np.where(
        np.logical_and.reduce(
            (
                cam_coord[2] > 0,
                projected_coord[0] / projected_coord[2] >= 0,
                projected_coord[0] / projected_coord[2] <= width // resolution - 1,
                projected_coord[1] / projected_coord[2] >= 0,
                projected_coord[1] / projected_coord[2] <= height // resolution - 1,
            )
        )
    )[0]
    pts_depths = cam_coord[-1:, valid_idx]
    projected_coord = projected_coord[:2, valid_idx] / projected_coord[-1:, valid_idx]
    depthmap[
        np.round(projected_coord[1]).astype(np.int32).clip(0, height // resolution - 1),
        np.round(projected_coord[0]).astype(np.int32).clip(0, width // resolution - 1),
    ] = pts_depths
    np.save(outputfile, depthmap)
    depthmap_normalized = (depthmap - np.min(depthmap)) / (np.max(depthmap) - np.min(depthmap))
    cv2.imwrite(f"test_{outputfile}.png", depthmap_normalized * 255)


if __name__ == "__main__":
    gen_depthmap("capoo1.ply", "calib_result1.yml", "0.npy")
    gen_depthmap("capoo2.ply", "calib_result2.yml", "1.npy")
    gen_depthmap("capoo.ply", "calib_result_projector.yml", "2.npy")
