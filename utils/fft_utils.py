import torch
import numpy as np
import cv2

_kernel_cache = {}


def get_fft_kernel(camera, channels=3, device="cuda"):
    global _kernel_cache
    key = (channels, camera.image_height, camera.image_width)
    w = camera.image_width
    h = camera.image_height

    R = (h + w) // 12
    if not key in _kernel_cache:
        hf_kernel = np.ones((h, w, channels), dtype=np.float32)

        hf_kernel = cv2.circle(hf_kernel, (w // 2, h // 2), R, (0, 0, 0), -1).transpose(2, 0, 1)

        _kernel_cache[key] = torch.from_numpy(hf_kernel).to(device=device)

    return _kernel_cache[key]
