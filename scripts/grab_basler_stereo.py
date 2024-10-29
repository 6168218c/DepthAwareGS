"""
insatll pypylon and opencv-python
run in cmd :
pip install  -i https://mirrors.zju.edu.cn/pypi/web/simple pypylon opencv-python
"""

from pypylon import pylon
from pypylon import genicam
import numpy as np
import sys
import cv2
import os
from datetime import datetime

# ================================================================================
# Grab images from the first camera,presss 's' to save a image, press 'q' to quit
# ================================================================================

default_cameraSettings = {
    # "r_balance": 1,
    # "g_balance": 1,
    # "b_balance": 1,
    "gain_db": 0,
    "exposure_time": 30000,
    "PixelFormat": "RGB8",
    "gamma": 1.0,
}


def OpenCameras():
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    if len(devices) == 0:
        return None
    cameras = [
        pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        for device in devices
    ]
    for camera in cameras:
        camera.Open()
    return cameras


def OpenCameraBySerialnumber(required_serial_number):
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    for device in devices:
        if device.GetSerialNumber() == required_serial_number:
            camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateDevice(device)
            )
            camera.Open()
            return camera
    return None


def SetCamera(camera, cameraSettings):
    # set whitebalance
    camera.PixelFormat.SetValue(cameraSettings["PixelFormat"])
    camera.UserSetSelector = "Default"
    camera.UserSetLoad.Execute()
    # camera.BalanceWhiteAuto.SetValue("Off")
    # camera.BalanceRatioSelector.SetValue = "Red"
    # camera.BalanceRatio.SetValue(cameraSettings["r_balance"])
    # camera.BalanceRatioSelector.SetValue = "Green"
    # camera.BalanceRatio.SetValue(cameraSettings["g_balance"])
    # camera.BalanceRatioSelector.SetValue = "Blue"
    # camera.BalanceRatio.SetValue(cameraSettings["b_balance"])
    # exposure time us
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(cameraSettings["exposure_time"])
    camera.GainAuto.SetValue("Off")
    camera.Gain.Value = cameraSettings["gain_db"]


def PrintInfo(camera):
    print("---------------------INFO OF CAMERA-----------------------")
    print("camera Model:", camera.GetDeviceInfo().GetModelName())
    print("Series_Number:", camera.GetDeviceInfo().GetSerialNumber())
    camera.BalanceRatioSelector.SetValue = "Red"
    print("White Balance R:", camera.BalanceRatio.Value)
    camera.BalanceRatioSelector.SetValue = "Green"
    print("White Balance G:", camera.BalanceRatio.Value)
    camera.BalanceRatioSelector.SetValue = "Blue"
    print("White Balance_B:", camera.BalanceRatio.Value)
    print("Gamma Value: ", camera.Gamma.Value)
    print("Exposure Time:", camera.ExposureTime.GetValue(), "us")
    print("Gain:", camera.Gain.Value, "dB")

    print("---------------------FINISH-----------------------")


if __name__ == "__main__":
    exitCode = 0
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # changed_parameter = (np.arange(10, 135, 5) * 1000).astype(np.int32)
    # changed_parameter_name = "exposure_time"
    save_root = os.path.join(os.getcwd(), f"Stereo_{folder_name}/")

    try:
        cameras = OpenCameras()
        if len(cameras) <= 0:
            print("WRONG,no camera connected")
        for camera in cameras:
            SetCamera(camera, default_cameraSettings)
            PrintInfo(camera)
        """The parameter MaxNumBuffer can be used to control the count of buffers
        allocated for grabbing. The default value of this parameter is 10."""
        # camera.MaxNumBuffer = 5

        """The camera device is parameterized with a default configuration which
        sets up free-running continuous acquisition."""
        # camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        converter = pylon.ImageFormatConverter()
        # converting to opencv bgr format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        save_index = 0
        while True:
            result_images = []
            for index, camera in enumerate(cameras):
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                if camera.IsGrabbing():
                    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                    grabResult: pylon.GrabResult = camera.RetrieveResult(
                        5000, pylon.TimeoutHandling_ThrowException
                    )
                    if grabResult.GrabSucceeded():
                        # Access the image data.
                        # img = cv2.cvtColor(grabResult.GetArray(), cv2.COLOR_BAYER_RG2RGB)
                        converted_img: pylon.PylonImage = converter.Convert(grabResult)
                        np_img = converted_img.GetArray()
                        result_images.append(np_img)
                    else:
                        print(
                            "Error: ", grabResult.ErrorCode, grabResult.ErrorDescription
                        )
                    grabResult.Release()
                camera.StopGrabbing()

            if len(result_images) != len(cameras):
                continue

            display_image = np.concatenate(
                [
                    cv2.resize(result_image, (640, 480))
                    for result_image in result_images
                ],
                axis=1,
            )
            cv2.imshow(
                "Grabbed Image",
                display_image,
            )
            os.makedirs(f"{save_root}", exist_ok=True)
            key = cv2.waitKey(0)

            if key == ord("s"):
                for index, result_image in enumerate(result_images):
                    os.makedirs(f"{save_root}{index}", exist_ok=True)
                    cv2.imwrite(
                        f"{save_root}{'%04d' % index}/{save_index}.png",
                        result_image,
                    )
                save_index += 1
            elif key == ord("q"):
                break
            else:
                cv2.destroyAllWindows()

    except genicam.GenericException as e:
        print("An exception occurred.")
        print(e)
        exitCode = 1
    sys.exit(exitCode)
