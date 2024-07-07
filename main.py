import cv2
from singleton_lock import SingletonLock
import tkinter as tk
from panel import CameraControlPanel

print_lock = SingletonLock.get_lock('print')

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


if __name__ == "__main__":
    main()