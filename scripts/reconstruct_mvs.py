import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import glob
import tqdm
from captured_data import (
    read_calib_data,
    decode_patterns,
    BASE_PATH,
    FIRST_SHOT_TARGET,
    SECOND_SHOT_TARGET,
    PROJ_RESOLUTION,
)


def write_ply(filename, vertices, faces=None, colors=None, cmap="BuGn"):
    """Store triangle mesh into .ply file format

    Writes a triangle mesh (optionally with per-vertex colors) into the ply file format, in a way consistent with, e.g., importing to Blender.

    Parameters
    ----------
    filename : str
        Name of the file ending in ".ply" to which to write the mesh
    vertices : numpy double array
        Matrix of mesh vertex coordinates
    faces : numpy int array, optional (default None)
        Matrix of triangle face indices into vertices. If none, only the vertices will be written (e.g., a point cloud)
    colors : numpy double array, optional (default None)
        Array of per-vertex colors. It can be a matrix of per-row RGB values, or a vector of scalar values that gets transformed by a colormap.
    cmap : str, optional (default 'BuGn')
        Name of colormap used to transform the color values if they are a vector of scalar function values (if colors is a matrix of RGB values, this parameter will not be used). Should be a valid input to `colormap`.

    See Also
    --------
    write_mesh, colormap.

    Notes
    -----
    This function is not optimized and covers the very specific funcionality of saving a mesh with per-vertex coloring that can be imported into Blender or other software.
    If you wish to write a mesh for any other purpose, we strongly recommend you use write_mesh instead.

    Examples
    --------
    TODO
    """

    vertices = vertices.astype(float)
    f = open(filename, "w")
    f.write("ply\nformat {} 1.0\n".format("ascii"))
    f.write("element vertex {}\n".format(vertices.shape[0]))
    f.write("property double x\n")
    f.write("property double y\n")
    f.write("property double z\n")
    if colors is not None:
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
    if faces is not None:
        f.write("element face {}\n".format(faces.shape[0]))
    # f.write("property list int int vertex_indices\n")
    f.write("end_header\n")
    # write_vert_str = "{} {} {}\n" * vertices.shape[0]
    # f.write(write_vert_str.format(tuple(np.reshape(vertices,(-1,1)))))
    # This for loop should be vectorized
    if colors is None:
        for i in range(vertices.shape[0]):
            f.write("{} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
    else:
        if np.max(colors) <= 1:
            C = np.round(colors * 255)
        else:
            C = colors
        # This should be vectorized
        for i in range(vertices.shape[0]):
            f.write(
                "{} {} {} {} {} {}\n".format(
                    vertices[i, 0],
                    vertices[i, 1],
                    vertices[i, 2],
                    int(C[i, 0]),
                    int(C[i, 1]),
                    int(C[i, 2]),
                )
            )
    # This should be vectorized
    if faces is not None:
        for i in range(faces.shape[0]):
            f.write("3 {} {} {}\n".format(faces[i, 0], faces[i, 1], faces[i, 2]))
    f.close()


def triangulate_stereo(K1, kc1, K2, kc2, Rt, T, p1, p2):
    """
    Triangulate 3D point from stereo image points.

    Parameters
    ----------
    K1 : numpy array
        Camera matrix for the first camera.
    kc1 : numpy array
        Distortion coefficients for the first camera.
    K2 : numpy array
        Camera matrix for the second camera.
    kc2 : numpy array
        Distortion coefficients for the second camera.
    Rt : numpy array
        Rotation matrix from the second camera to the first camera.
    T : numpy array
        Translation vector from the second camera to the first camera.
    p1 : tuple
        Image point in the first camera.
    p2 : tuple
        Image point in the second camera.

    Returns
    -------
    p3d : numpy array
        Triangulated 3D point.
    distance : float
        Distance between the rays.
    """
    # to image camera coordinates
    # assume p1 and p2 are [N, 2]
    p1 = cv2.undistortPoints(p1, K1, kc1).squeeze().T
    p2 = cv2.undistortPoints(p2, K2, kc2).squeeze().T
    u1 = np.vstack([p1, np.ones(p1.shape[1])])
    u2 = np.vstack([p2, np.ones(p2.shape[1])])

    # to world coordinates
    w1 = u1
    w2 = Rt @ (u2 - T)

    # world rays
    v1 = w1
    v2 = Rt @ u2

    # compute ray-ray approximate intersection
    p3d, distance = approximate_ray_intersection(v1, w1, v2, w2)
    return p3d, distance


def approximate_ray_intersection(v1, q1, v2, q2):
    """
    Approximate the intersection of two rays.

    Parameters
    ----------
    v1 : numpy array 3 * N
        Direction vector of the first ray.
    q1 : numpy array 3 * N
        Point on the first ray.
    v2 : numpy array 3 * N
        Direction vector of the second ray.
    q2 : numpy array 3 * N
        Point on the second ray.

    Returns
    -------
    p : numpy array
        Approximate intersection point.
    distance : float
        Distance between the rays.
    """
    v1tv1 = np.sum(v1 * v1, axis=0)
    v2tv2 = np.sum(v2 * v2, axis=0)
    v1tv2 = np.sum(v1 * v2, axis=0)
    v2tv1 = np.sum(v2 * v1, axis=0)

    detV = v1tv1 * v2tv2 - v1tv2 * v2tv1

    q2_q1 = q2 - q1
    Q1 = np.sum(v1 * q2_q1, axis=0)
    Q2 = -np.sum(v2 * q2_q1, axis=0)

    lambda1 = (v2tv2 * Q1 + v1tv2 * Q2) / detV
    lambda2 = (v2tv1 * Q1 + v1tv1 * Q2) / detV

    p1 = lambda1 * v1 + q1
    p2 = lambda2 * v2 + q2

    p = 0.5 * (p1 + p2)
    distance = np.linalg.norm(p2 - p1, axis=0)

    return p.T, distance


def pattern_match_patched(target_folder, calib_file):
    """
    Match patterns and build dense point cloud using triangulation
    Args:
        target_folder (str): target folder for reconstruction
    """
    decoded_images = decode_patterns([target_folder])
    image_files = glob.glob(os.path.join(target_folder, "*.png"))
    image_files.sort()
    white_image = cv2.imread(image_files[0])

    assert len(decoded_images) == 1
    pattern_image, mask = decoded_images[0]

    calib_data = read_calib_data(calib_file)
    cam_mtx, cam_dist, proj_mtx, proj_dist, R, T = (
        calib_data["cam_mtx"],
        calib_data["cam_dist"],
        calib_data["proj_mtx"],
        calib_data["proj_dist"],
        calib_data["R"],
        calib_data["T"],
    )

    # filter points with valid patterns
    valid_coords = np.argwhere(mask > 0)
    cam_points = valid_coords[:, ::-1]
    proj_points = pattern_image[mask > 0].astype(np.int32)

    white_image = cv2.undistort(white_image, cam_mtx, cam_dist)
    white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB)
    point_colors = white_image[mask > 0].astype(np.float32) / 255.0

    proj_patch_coords = np.zeros((*PROJ_RESOLUTION[::-1], 2), dtype=np.float32)
    proj_patch_colors = np.zeros((*PROJ_RESOLUTION[::-1], 3), dtype=np.float32)
    proj_patch_cnt = np.zeros(PROJ_RESOLUTION[::-1], dtype=np.int32)
    for i in tqdm.trange(len(cam_points), desc="Matching points"):
        cam_point, proj_point, point_color = (
            cam_points[i],
            proj_points[i],
            point_colors[i],
        )
        proj_patch_coords[proj_point[1], proj_point[0]] += cam_point
        proj_patch_colors[proj_point[1], proj_point[0]] += point_color
        proj_patch_cnt[proj_point[1], proj_point[0]] += 1

    proj_points = np.argwhere(proj_patch_cnt > 0)[:, ::-1].astype(np.float32)
    cam_points = (
        proj_patch_coords[proj_patch_cnt > 0]
        / proj_patch_cnt[proj_patch_cnt > 0][:, None]
    )
    point_colors = (
        (
            proj_patch_colors[proj_patch_cnt > 0]
            / proj_patch_cnt[proj_patch_cnt > 0][:, None]
        )
        * 255.0
    ).astype(np.uint8)

    # undistort points into identity camera
    # cam_points = cv2.undistortPoints(cam_points, cam_mtx, cam_dist, P=cam_mtx).squeeze()
    # proj_points = cv2.undistortPoints(
    #     proj_points, proj_mtx, proj_dist, P=proj_mtx
    # ).squeeze()

    # triangulate points(see cv2.stereoCalibrate for R and T)
    projector2camera = np.zeros((4, 4), dtype=np.float32)
    projector2camera[:3, :3] = R
    projector2camera[:3, 3] = T.squeeze()
    projector2camera[3, 3] = 1
    camera2projector = np.linalg.inv(projector2camera)

    # Now projector is the coordinate origin
    ## TODO Obtain camera R,T inside projector coordinate

    cam_world2proj = cam_mtx @ camera2projector[:3]
    proj_world2proj = proj_mtx @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # point_cloud = cv2.triangulatePoints(
    #     cam_world2proj,
    #     proj_world2proj,
    #     cam_points.T,
    #     proj_points.T,
    # )
    point_cloud, distances = triangulate_stereo(
        cam_mtx, cam_dist, proj_mtx, proj_dist, R.T, T, cam_points, proj_points
    )
    # point_cloud = (point_cloud[:3] / point_cloud[3]).T
    point_cloud = (  # transport coordinates from camera to projector
        R @ point_cloud.T + T
    ).T
    # point_cloud = np.vstack([point_cloud, [0, 0, 0]])  # get the coords origin position
    # colors = np.vstack([colors, [0, 0, 0]])

    threshold = 1
    point_cloud = point_cloud[distances < threshold]
    colors = point_colors[distances < threshold]
    proj_points = proj_points[distances < threshold]

    return (point_cloud, colors, proj_points)


def multi_view_stereo(target_folders, calib_files, outfile):
    """
    Match patterns and build dense point cloud using triangulation
    Args:
        target_folder (str): target folder for reconstruction
    """

    point_clouds = []
    colors = []
    proj_points = []
    for target_folder, calib_file in zip(target_folders, calib_files):
        point_cloud, color, proj_point = pattern_match_patched(
            target_folder, calib_file
        )
        point_clouds.append(point_cloud)
        colors.append(color)
        proj_points.append(proj_point)

    point_cloud = np.concatenate(point_clouds)
    colors = np.concatenate(colors)
    proj_points = np.concatenate(proj_points).astype(np.int32)

    pcd_voxel_coords = np.zeros((*PROJ_RESOLUTION[::-1], 3), dtype=np.float32)
    pcd_voxel_colors = np.zeros((*PROJ_RESOLUTION[::-1], 3), dtype=np.float32)
    pcd_voxel_cnt = np.zeros(PROJ_RESOLUTION[::-1], dtype=np.int32)
    for i in tqdm.trange(len(point_cloud), desc="Matching points"):
        coords, color, proj_point = (point_cloud[i], colors[i], proj_points[i])
        pcd_voxel_coords[proj_point[1], proj_point[0]] += coords
        pcd_voxel_colors[proj_point[1], proj_point[0]] += color
        pcd_voxel_cnt[proj_point[1], proj_point[0]] += 1

    valid_points = pcd_voxel_cnt > 0
    point_cloud = pcd_voxel_coords[valid_points] / pcd_voxel_cnt[valid_points, None]
    colors = pcd_voxel_colors[valid_points] / pcd_voxel_cnt[valid_points, None]

    import open3d as o3d

    cloud_visual = o3d.geometry.PointCloud()
    cloud_visual.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(point_cloud, dtype=np.float64)
    )
    cloud_visual.colors = o3d.utility.Vector3dVector(
        np.ascontiguousarray(colors / 255.0, dtype=np.float64)
    )
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(cloud_visual)
    visualizer.run()
    visualizer.destroy_window()
    write_ply(outfile, point_cloud, colors=colors)


def main():
    multi_view_stereo(
        [
            os.path.join(BASE_PATH, FIRST_SHOT_TARGET),
            os.path.join(BASE_PATH, SECOND_SHOT_TARGET),
        ],
        ["calib_result1.yml", "calib_result2.yml"],
        "capoo.ply",
    )


if __name__ == "__main__":
    main()
