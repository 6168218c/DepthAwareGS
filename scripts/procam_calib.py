import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import glob
from captured_data import (
    decode_patterns,
    write_calib_data,
    PROJ_RESOLUTION,
    BASE_PATH,
    FIRST_SHOT_NAMES,
    FIRST_SHOT_TARGET,
)

CHESS_SIZE = (8, 11)
CHESS_OBJPOINTS = np.zeros((CHESS_SIZE[0] * CHESS_SIZE[1], 3), np.float32)
CHESS_OBJPOINTS[:, :2] = np.mgrid[0 : CHESS_SIZE[0], 0 : CHESS_SIZE[1]].T.reshape(-1, 2)


def calibrate_camera(decoded_images, captured_pattern_folders, window_radius=None):
    objpoints = []
    cam_imagepoints = []
    proj_imagepoints = []

    for i, folder in enumerate(captured_pattern_folders):
        image_files = glob.glob(os.path.join(folder, "*.png"))
        image_files.sort()

        white_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        decoded_image, mask = decoded_images[i]

        if window_radius is None:
            window_radius = 60
        print("Using window size: ", window_radius * 2 + 1)

        res, corners = cv2.findChessboardCorners(white_image, CHESS_SIZE, None)
        if not res:
            print("Chessboard not found in ", folder)
            continue

        corners = cv2.cornerSubPix(
            white_image, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        )

        objpoints.append(CHESS_OBJPOINTS)
        cam_imagepoints.append(corners)

        proj_corners = []
        for corner in corners:
            x, y = np.round(corner).astype(np.int32).ravel()

            ymin = max(y - window_radius, 0)
            ymax = min(y + window_radius + 1, white_image.shape[0])
            xmin = max(x - window_radius, 0)
            xmax = min(x + window_radius + 1, white_image.shape[1])

            src_window = np.mgrid[ymin:ymax, xmin:xmax].T
            pattern_window = decoded_image[ymin:ymax, xmin:xmax]
            mask_window = mask[ymin:ymax, xmin:xmax]

            src_points = src_window[mask_window > 0][:, ::-1]
            dst_points = pattern_window[mask_window > 0]

            if len(src_points) < window_radius**2:
                print(f"[WARN] Too few corners found in {x} {y} in folder {folder}")

            H, inliners = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
            Q = H @ np.array([*corner.ravel(), 1]).transpose()
            q = Q[0:2] / Q[2]
            proj_corners.append(q)
        proj_imagepoints.append(np.array(proj_corners, dtype=corners.dtype))

    flags = cv2.CALIB_FIX_K3
    cam_error, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
        objpoints,
        cam_imagepoints,
        white_image.shape[::-1],
        None,
        None,
        None,
        None,
        flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, np.finfo(np.float64).eps),
    )
    print("Camera calibration RMS: ", cam_error)
    proj_error, proj_mtx, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        objpoints,
        proj_imagepoints,
        PROJ_RESOLUTION,
        None,
        None,
        None,
        None,
        flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, np.finfo(np.float64).eps),
    )
    print("Projector calibration RMS: ", proj_error)
    stereo_error, cam_mtx, cam_dist, proj_mtx, proj_dist, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        cam_imagepoints,
        proj_imagepoints,
        cam_mtx,
        cam_dist,
        proj_mtx,
        proj_dist,
        white_image.shape[::-1],
        None,
        None,
        None,
        None,
        flags + cv2.CALIB_FIX_INTRINSIC,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 150, np.finfo(np.float64).eps),
    )
    print("Stereo calibration RMS: ", stereo_error)

    write_calib_data(
        cam_mtx,
        cam_dist,
        proj_mtx,
        proj_dist,
        R,
        T,
        cam_error,
        proj_error,
        stereo_error,
        white_image.shape[1],
        white_image.shape[0],
        filename="calib_result1.yml",
    )

    write_calib_data(
        proj_mtx,
        proj_dist,
        proj_mtx,
        proj_dist,
        np.diag(3),
        np.zeros((3, 1)),
        cam_error,
        proj_error,
        stereo_error,
        PROJ_RESOLUTION[0],
        PROJ_RESOLUTION[1],
        "calib_result_projector.yml",
    )


def main():
    captured_pattern_folders = set([os.path.join(BASE_PATH, foldername) for foldername in FIRST_SHOT_NAMES])
    target_folder = os.path.join(BASE_PATH, FIRST_SHOT_TARGET)
    captured_pattern_folders = list(captured_pattern_folders)
    captured_pattern_folders.sort()

    print("Running on captured patterns: ", captured_pattern_folders)

    decoded_images = decode_patterns([*captured_pattern_folders, target_folder])

    calibrate_camera(decoded_images, captured_pattern_folders)


if __name__ == "__main__":
    main()
