import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate, print_cam_informations

class CameraReader(Process):
    def __init__(self, cam_id, config, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"CameraReader {self.cam_id}"
        self.config = config
        self.queue = Queue(maxsize = 5)
        self.shared_dict = shared_dict
        self.shared_dict[self.process_name] = {
            'fps': 0,
            'running': True
        }

    def run(self):
        tprint(f"Start camera {self.cam_id}")
        try:
            self.read_camera()
        except Exception as e:
            tprint(f"Exception in camera {self.cam_id}: {e}")
        finally:
            # queue cancel_join_thread() to prevent block join_thread()
            # https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues
            self.queue.close()
            self.queue.cancel_join_thread()
            tprint(f"Stopping camera {self.cam_id}")

    def read_camera(self):
        frameWidth, frameHeight, fps = decode_frame_size_rate(self.config['resolution_fps_setting'])

        cam = cv2.VideoCapture(self.cam_id, self.config['api'])
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
        cam.set(cv2.CAP_PROP_FPS, fps)
        cam.set(cv2.CAP_PROP_EXPOSURE, self.config['exposure'])
        cam.set(cv2.CAP_PROP_GAIN, self.config['gain'])

        if cam.isOpened():  # try to get the first frame
            rval, frame = cam.read()
        else:
            rval = False
        with print_lock:
            print_cam_informations(self.cam_id, cam)

        prev_time = time.time()
        times = [1]
        maxCount = 60
        while rval and self.shared_dict[self.process_name]['running']:
            current_time = time.time()
            times.append(round(current_time - prev_time, 2))
            times = times[-maxCount:]
            status = self.shared_dict[self.process_name]
            status['fps'] = 1 / np.mean(times) #計算時間
            self.shared_dict[self.process_name] = status
            prev_time = current_time

            rval, frame = cam.read()
            if self.queue.full():
                self.queue.get()
            self.queue.put(frame)
            # tprint(f"fps: {round(self.shared_dict[self.process_name]['fps'])}")
            # else:
            #     tprint(f"queue {self.cam_id} full!")
        status = self.shared_dict[self.process_name]
        status['running'] = False
        self.shared_dict[self.process_name] = status
        cam.release()
        tprint(f"Released camera {self.cam_id}")