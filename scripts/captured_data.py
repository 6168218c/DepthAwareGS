"""Captured Data
Global defines for data processing
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import tqdm
import glob

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Images", "structured_light")
FIRST_SHOT_NAMES = ["1", "2", "3"]
SECOND_SHOT_NAMES = ["4"]

FIRST_SHOT_TARGET = "capoo1"
SECOND_SHOT_TARGET = "capoo2"

BLACK_THRESHOLD = 15
WHITE_THRESHOLD = 5

PROJ_RESOLUTION = (512, 384)


def decode_patterns(captured_pattern_folders: list[str], force_refresh=False):
    graycode = cv2.structured_light.GrayCodePattern.create(*PROJ_RESOLUTION)
    graycode.setWhiteThreshold(WHITE_THRESHOLD)
    graycode.setBlackThreshold(BLACK_THRESHOLD)

    def compute_shadow_mask(black_image, white_image, threshold):
        shadow_mask = np.zeros_like(black_image)
        shadow_mask[white_image > black_image + threshold] = 255
        return shadow_mask

    def decode_pixels(captured_patterns, mask):
        decoded_image = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.float32)
        valid_points = np.argwhere(mask > 0)
        real_mask = mask.copy()
        extra_mask_count = 0
        for j, i in tqdm.tqdm(valid_points, desc="Decoding patterns..."):
            error, proj_pixel = graycode.getProjPixel(captured_patterns, i, j)
            if error:
                real_mask[j, i] = 0
                extra_mask_count += 1
                continue
            decoded_image[j, i, 0] = float(proj_pixel[0])
            decoded_image[j, i, 1] = float(proj_pixel[1])

        print(f"Additionally masked {extra_mask_count} points")
        return decoded_image, real_mask

    decoded_images = []
    for folder in captured_pattern_folders:
        DECODE_SAVE_PATH = os.path.join(os.path.dirname(folder), f"pattern_{os.path.basename(folder)}.npy")
        MASK_SAVE_PATH = os.path.join(os.path.dirname(folder), f"mask_{os.path.basename(folder)}.png")
        if not force_refresh and os.path.exists(DECODE_SAVE_PATH) and os.path.exists(MASK_SAVE_PATH):
            decoded_images.append((np.load(DECODE_SAVE_PATH), cv2.imread(MASK_SAVE_PATH, cv2.IMREAD_GRAYSCALE)))
            continue
        image_files = glob.glob(os.path.join(folder, "*.png"))
        image_files.sort()

        images = [cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) for image_file in image_files]

        black_image = images[1]
        white_image = images[0]
        captured_patterns = images[2:]

        mask = compute_shadow_mask(black_image, white_image, BLACK_THRESHOLD)
        decoded_image, mask = decode_pixels(captured_patterns, mask)
        np.save(DECODE_SAVE_PATH, decoded_image)
        cv2.imwrite(MASK_SAVE_PATH, mask)

        decoded_images.append((decoded_image, mask))

    return decoded_images


def write_calib_data(
    cam_mtx,
    cam_dist,
    proj_mtx,
    proj_dist,
    R,
    T,
    cam_error,
    proj_error,
    stereo_error,
    width,
    height,
    filename="calib_result.yml",
):
    fs = cv2.FileStorage(os.path.join(BASE_PATH, filename), cv2.FileStorage_WRITE)
    fs.write("cam_K", cam_mtx)
    fs.write("cam_kc", cam_dist)
    fs.write("proj_K", proj_mtx)
    fs.write("proj_kc", proj_dist)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("cam_error", cam_error)
    fs.write("proj_error", proj_error)
    fs.write("stereo_error", stereo_error)
    fs.write("width", width)
    fs.write("height", height)


def read_calib_data(filename="calib_result.yml"):
    ARCHIVE_PATH = os.path.join(BASE_PATH, filename)
    if not os.path.exists(ARCHIVE_PATH):
        raise FileNotFoundError(ARCHIVE_PATH)

    fs = cv2.FileStorage(ARCHIVE_PATH, cv2.FileStorage_READ)

    return {
        "cam_mtx": fs.getNode("cam_K").mat(),
        "cam_dist": fs.getNode("cam_kc").mat(),
        "proj_mtx": fs.getNode("proj_K").mat(),
        "proj_dist": fs.getNode("proj_kc").mat(),
        "R": fs.getNode("R").mat(),
        "T": fs.getNode("T").mat(),
        "height": fs.getNode("height").real(),
        "width": fs.getNode("width").real(),
    }
