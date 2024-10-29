import cv2
import numpy as np
import os

PROJ_RESOLUTION = (512, 384)


def main():
    graycode = cv2.structured_light.GrayCodePattern.create(*PROJ_RESOLUTION)
    ret, patterns = graycode.generate()
    blackImage = np.zeros_like(patterns[0])
    whiteImage = np.zeros_like(patterns[0])
    blackImage, whiteImage = graycode.getImagesForShadowMasks(blackImage, whiteImage)
    patterns = [blackImage, whiteImage, *patterns]

    os.makedirs("patterns", exist_ok=True)
    for i, pattern in enumerate(patterns):
        cv2.imwrite("patterns/%02d.png" % (i + 1), pattern)


if __name__ == "__main__":
    main()
