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
import torch
import open3d as o3d


def corrected_decode(
    captured_pattern_folders: list[str],
    calib_data,
    force_refresh=False,
):
    cam_mtx, cam_dist = calib_data["cam_mtx"], calib_data["cam_dist"]

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
        DECODE_SAVE_PATH = os.path.join(
            os.path.dirname(folder), f"corrected_pattern_{os.path.basename(folder)}.npy"
        )
        MASK_SAVE_PATH = os.path.join(
            os.path.dirname(folder), f"corrected_mask_{os.path.basename(folder)}.png"
        )
        if (
            not force_refresh
            and os.path.exists(DECODE_SAVE_PATH)
            and os.path.exists(MASK_SAVE_PATH)
        ):
            decoded_images.append(
                (
                    np.load(DECODE_SAVE_PATH),
                    cv2.imread(MASK_SAVE_PATH, cv2.IMREAD_GRAYSCALE),
                    cv2.undistort(
                        cv2.imread(
                            glob.glob(os.path.join(folder, "*.png"))[0],
                        ),
                        cam_mtx,
                        cam_dist,
                    ),
                )
            )
            continue
        image_files = glob.glob(os.path.join(folder, "*.png"))
        image_files.sort()

        images = [
            cv2.undistort(
                cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), cam_mtx, cam_dist
            )
            for image_file in image_files
        ]

        black_image = images[1]
        white_image = images[0]
        captured_patterns = images[2:]

        mask = compute_shadow_mask(black_image, white_image, BLACK_THRESHOLD)
        decoded_image, mask = decode_pixels(captured_patterns, mask)
        np.save(DECODE_SAVE_PATH, decoded_image)
        cv2.imwrite(MASK_SAVE_PATH, mask)

        decoded_images.append((decoded_image, mask, white_image))

    return decoded_images


def get_projector_view(decoded_tuple):
    pattern, mask, white = decoded_tuple
    pattern = pattern.astype(np.int32)
    labels = pattern[mask > 0]
    labels = labels[:, 1] * PROJ_RESOLUTION[0] + labels[:, 0]
    labels = torch.from_numpy(labels).to(torch.int64)
    labels = labels.reshape(-1, 1).repeat(1, 3)

    src_image = torch.from_numpy(white[mask > 0]).to(torch.float32)

    view = torch.zeros(
        (PROJ_RESOLUTION[0] * PROJ_RESOLUTION[1], 3), dtype=torch.float32
    )
    count = torch.zeros(PROJ_RESOLUTION[0] * PROJ_RESOLUTION[1], dtype=torch.int32)
    view.scatter_add_(
        0,
        labels,
        src_image.reshape(-1, 3),
    )
    count.scatter_add_(
        0, labels[:, 0], torch.ones_like(src_image[:, 0], dtype=torch.int32)
    )
    view[count > 0] /= count[count > 0].reshape(-1, 1)
    view = (
        view.reshape(PROJ_RESOLUTION[1], PROJ_RESOLUTION[0], 3).numpy().astype(np.uint8)
    )

    return view


def get_view_no_coordinates():
    import matplotlib.pyplot as plt

    calib_data = read_calib_data("calib_result1.yml")
    decoded_1 = corrected_decode(
        [os.path.join(BASE_PATH, FIRST_SHOT_TARGET)], calib_data
    )
    calib_data = read_calib_data("calib_result2.yml")
    decoded_2 = corrected_decode(
        [os.path.join(BASE_PATH, SECOND_SHOT_TARGET)], calib_data
    )

    view_1 = get_projector_view(decoded_1[0])
    view_2 = get_projector_view(decoded_2[0])

    mask_1 = view_1.sum(axis=-1) > 0
    mask_2 = view_2.sum(axis=-1) > 0
    overlapped = np.logical_and(mask_1, mask_2)

    final_view = view_1.astype(np.int32) + view_2.astype(np.int32)
    final_view[overlapped] //= 2

    b, g, r = cv2.split(final_view.astype(np.uint8))
    a = np.logical_or(mask_1, mask_2).astype(np.uint8) * 255
    final_view = cv2.merge((b, g, r, a))

    # calib_data = read_calib_data("calib_result_projector.yml")
    # cam_mtx, cam_dist, width, height = (
    #     calib_data["cam_mtx"],
    #     calib_data["cam_dist"],
    #     calib_data["width"],
    #     calib_data["height"],
    # )
    # new_mtx = cam_mtx.copy()
    # new_mtx[:2, 2] = [width // 2, height // 2]
    # final_view = cv2.undistort(final_view, cam_mtx, cam_dist, newCameraMatrix=new_mtx)
    cv2.imwrite("final_view.png", final_view)


if __name__ == "__main__":
    get_view_no_coordinates()
