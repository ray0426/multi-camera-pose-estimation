import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate, print_cam_informations

class CameraReader(Process):
    def __init__(self, cam_id, config, original_image, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"CameraReader {self.cam_id}"
        self.config = config
        self.original_image = original_image
        self.shared_dict = shared_dict
        self.shared_dict[self.process_name] = {
            'fps': 0,
            'running': True,
            'image_idx': 0
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
            # self.queue.close()
            # self.queue.cancel_join_thread()
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
        image_idx = 0
        while rval and not self.shared_dict["control signals"][self.process_name]["halt"]:
            current_time = time.time()
            times.append(round(current_time - prev_time, 2))
            times = times[-maxCount:]
            local_dict = self.shared_dict[self.process_name]
            local_dict['fps'] = 1 / np.mean(times) #計算時間
            local_dict['image_idx'] = image_idx
            prev_time = current_time
            self.shared_dict[self.process_name] = local_dict

            rval, frame = cam.read()
            original_image = np.frombuffer(self.original_image.get_obj(), dtype=np.uint8).reshape((frameHeight, frameWidth, 3))
            original_image[:] = frame
            image_idx = image_idx + 1
        local_dict = self.shared_dict[self.process_name]
        local_dict['running'] = False
        self.shared_dict[self.process_name] = local_dict
        cam.release()
        tprint(f"Released camera {self.cam_id}")