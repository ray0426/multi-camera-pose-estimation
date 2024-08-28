import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate

class CameraDisplayer(Process):
    def __init__(self, cam_id, config, original_image, input_camera_name, input_hpe_name, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"CameraDisplayer {self.cam_id}"
        self.screen_name = f"camera {cam_id}"
        self.config = config
        self.input_camera_name = input_camera_name
        self.input_hpe_name = input_hpe_name
        self.original_image = original_image
        self.shared_dict = shared_dict
        self.shared_dict[self.process_name] = {
            'fps': 0,
            'running': True
        }

    def run(self):
        tprint("Displaying camera" + str(self.cam_id))
        try:
            self.display_camera()
        except Exception as e:
            tprint(f"Exception in camera {self.cam_id}: {e}")
        finally:
            tprint(f"Stopping camera {self.cam_id}")
            cv2.destroyWindow(self.screen_name)
    
    def display_camera(self):
        frameWidth, frameHeight, fps = decode_frame_size_rate(self.config['resolution_fps_setting'])
        cv2.namedWindow(self.screen_name)

        prev_time = time.time()
        times = [1]
        maxCount = 60
        prev_image_idx = -1
        while not self.shared_dict["control signals"][self.process_name]["halt"]:
            input_frame = np.frombuffer(self.original_image.get_obj(), dtype=np.uint8).reshape((frameHeight, frameWidth, 3))
            if (input_frame is not None) and not (prev_image_idx == self.shared_dict[self.input_camera_name]['image_idx']):
                prev_image_idx = self.shared_dict[self.input_camera_name]['image_idx']
                current_time = time.time()
                times.append(round(current_time - prev_time, 2))
                times = times[-maxCount:]
                status = self.shared_dict[self.process_name]
                status['fps'] = 1 / np.mean(times) #計算時間
                self.shared_dict[self.process_name] = status
                prev_time = current_time

                frame = input_frame
                cv2.putText(
                    frame, 'FPS : {0:.2f}'.format(round(self.shared_dict[self.process_name]['fps'], 2)), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(
                    frame, 'Time : {0:.2f}'.format(round(1 / self.shared_dict[self.process_name]['fps'], 5)), 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(self.screen_name, frame)
                # 按下 'q' 鍵退出
                if cv2.waitKey(1) == ord('q'):
                    status = self.shared_dict[self.process_name]
                    status['running'] = False
                    self.shared_dict[self.process_name] = status
                    break
            else:
                # tprint(f"queue {self.cam_id} empty!")
                # time.sleep(0.005)
                pass