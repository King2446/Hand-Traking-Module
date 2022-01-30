import os


def CheckNow ():
    try:
        import cv2, mediapipe, numpy, math, time

    except:
        pkgs = ["opencv-python", "mediapipe", "numpy"]

        for pkg in pkgs:
            os.system (f"pip3 install {pkg}")


if __name__ == "__main__":
    CheckNow ()


# THE END
