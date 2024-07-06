import sys
import os
import cv2
import queue
import threading
from singleton_lock import SingletonLock
from read_camera import asyn_open_cameras, stop_all_threads
import tkinter as tk
from panel import CameraControlPanel

print_lock = SingletonLock.get_lock('print')

def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    else:
        return gettrace() is not None

def load_openpose_module():
    # path to openpose repo directory
    OPENPOSE_PATH = "D:/coding/pose_estimation_projects/openpose/"
    # path to build directory
    BUILD_PATH = "build/python37/"
    # load builded files according to Debug/Release mode
    if is_debugging():
        print("Currently running in debug mode.")
        BUILD_TYPE = "Debug/"
    else:
        print("Not running in debug mode.")
        BUILD_TYPE = "Release/"
    # pyd file
    sys.path.append(OPENPOSE_PATH + BUILD_PATH + 'python/openpose/' + BUILD_TYPE)
    # openpose.dll
    os.environ['PATH']  = os.environ['PATH'] + ';' + OPENPOSE_PATH + BUILD_PATH + 'x64/' + BUILD_TYPE
    # other dll files
    os.environ['PATH']  = os.environ['PATH'] + ';' +  OPENPOSE_PATH + BUILD_PATH + 'bin/;'
    try:
        import pyopenpose as op # type: ignore
    except Exception as e:
        print(e)
        exit(-1)

def main():
    print('\n==============Start program==============')
    config = {
        # 'resolution_fps_setting': "640x480@30",
        'resolution_fps_setting': "1280x720@60",
        # 'resolution_fps_setting': "1920x1080@30",
        'api': cv2.CAP_MSMF,
        'exposure': -7,
        'gain': 200,
    }

    root = tk.Tk()
    root.title("Main Application")
    app = CameraControlPanel(
        master = root, 
        config = config, 
        camera_ids = [0, 1]
    )
    app.pack()
    root.mainloop()

    # load_openpose_module()

if __name__ == "__main__":
    main()