import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate

class CameraDisplayer(Process):
    def __init__(self, cam_id, config, original_image, pose_2d, input_camera_name, input_hpe_name, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"CameraDisplayer {self.cam_id}"
        self.screen_name = f"camera {cam_id}"
        self.config = config
        self.original_image = original_image
        self.pose_2d = pose_2d
        self.input_camera_name = input_camera_name
        self.input_hpe_name = input_hpe_name
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
                pose_2d = np.frombuffer(self.pose_2d.get_obj(), dtype=np.float32).reshape((25, 3))
                frame = self.draw_human_2d(frame, pose_2d)
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
    
    def draw_human_2d(self, frame, pose_2d):
        '''
            ---1080--- x
         |
        720
         |
         y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 7)   # BGR
        '''
        BODY25_SKELETON_EDGES = np.array([
            [0, 1],
            [1, 2], [2, 3], [3, 4],
            [1, 5], [5, 6], [6, 7],
            [1, 8],
            [8, 9], [9, 10], [10, 11], [11, 22], [11, 24], [22, 23],
            [8, 12], [12, 13], [13, 14], [14, 19], [14, 21], [19, 20],
            [0, 15], [15, 17],
            [0, 16], [16, 18]
        ])
        for i in range(len(BODY25_SKELETON_EDGES)):
            start_idx = BODY25_SKELETON_EDGES[i, 0]
            end_idx = BODY25_SKELETON_EDGES[i, 1]
            x1, y1, c1 = pose_2d[start_idx]
            x2, y2, c2 = pose_2d[end_idx]
            if c1 > 0.15 and c2 > 0.15:
                if i % 2 == 1:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (238, 212, 165), 7)
                else:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (213, 175, 235), 7)
            
        return frame