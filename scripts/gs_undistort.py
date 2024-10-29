import cv2
import numpy as np
import os
import tqdm
import glob
from captured_data import (
    PROJ_RESOLUTION,
    BASE_PATH,
    WHITE_THRESHOLD,
    BLACK_THRESHOLD,
)
from captured_data import (
    read_calib_data,
    BASE_PATH,
    FIRST_SHOT_TARGET,
    SECOND_SHOT_TARGET,
)

if __name__ == "__main__":
    calib_data = read_calib_data("calib_result1.yml")
    cam_mtx, cam_dist, width, height = (
        calib_data["cam_mtx"],
        calib_data["cam_dist"],
        calib_data["width"],
        calib_data["height"],
    )

    image = cv2.imread(os.path.join(BASE_PATH, FIRST_SHOT_TARGET, "0000.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    new_mtx = cam_mtx.copy()
    # new_mtx[:2, 2] = [width // 2, height // 2]
    image = cv2.undistort(image, cam_mtx, cam_dist)
    cv2.imwrite("0.png", image)
