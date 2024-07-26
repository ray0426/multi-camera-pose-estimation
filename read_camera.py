import cv2
import numpy as np
import time
from multiprocessing import Process, Queue
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate

def open_single_camera(cam_id, config):
    frameWidth, frameHeight, fps = decode_frame_size_rate(config['resolution_fps_setting'])
    cap = cv2.VideoCapture(cam_id, config['api'])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print("無法打開相機")
        exit()

    times = [1]
    maxCount = 20
    prev_time = time.time()
    while True:
        # 從相機讀取一幀
        ret, frame = cap.read()

        # 如果讀取失敗
        if not ret:
            print("無法獲取影像")
            break

        # 計算fps相關
        current_time = time.time()
        times.append(round(current_time - prev_time, 2))
        times = times[-maxCount:]
        optime = np.mean(times) 
        prev_time = current_time
        cv2.putText(frame,'FPS : {0:.2f}'.format(round(1/optime,2)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame,'Time : {0:.2f}'.format(round(optime, 5)), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 顯示影像
        cv2.imshow('Camera', frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 釋放相機資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

def syn_open_multiple_cameras(cam_ids, config):
    frameWidth, frameHeight, fps = decode_frame_size_rate(config['resolution_fps_setting'])
    # 打開相機
    cap0 = cv2.VideoCapture(cam_ids[0], config['api'])
    cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap0.set(cv2.CAP_PROP_FPS, fps)
    cap0.set(cv2.CAP_PROP_EXPOSURE, config['exposure'])
    cap0.set(cv2.CAP_PROP_GAIN, config['gain'])
    cap1 = cv2.VideoCapture(cam_ids[1], config['api'])
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap1.set(cv2.CAP_PROP_FPS, fps)
    cap1.set(cv2.CAP_PROP_EXPOSURE, config['exposure'])
    cap1.set(cv2.CAP_PROP_GAIN, config['gain'])

    if not cap0.isOpened() or not cap1.isOpened():
        print("無法打開相機")
        exit()

    # 紀錄每frame時間
    times = [1]
    maxCount = 20
    prev_time = time.time()
    while True:
        # 從兩個相機讀取幀
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        # 如果讀取失敗
        if not ret0 or not ret1:
            print("無法獲取影像")
            break

        # 調整兩個幀的大小一致
        frame0 = cv2.resize(frame0, (frameWidth, frameHeight))
        frame1 = cv2.resize(frame1, (frameWidth, frameHeight))

        # 拼接兩個幀
        combined_frame = np.hstack((frame0, frame1))

        # 計算fps相關
        current_time = time.time()
        times.append(round(current_time - prev_time, 2))
        times = times[-maxCount:]
        optime = np.mean(times) 
        prev_time = current_time
        cv2.putText(combined_frame,'FPS : {0:.2f}'.format(round(1/optime,2)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame,'Time : {0:.2f}'.format(round(optime, 5)), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 顯示拼接後的幀
        cv2.imshow('Combined Camera', combined_frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 釋放相機資源並關閉視窗
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def print_cam_informations(cam_id, cam, type="simple"):
    with print_lock:
        print("=============" + "camera " + str(cam_id) + "=============")
        print("FPS :", cam.get(cv2.CAP_PROP_FPS))
        print("Frame size :", cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if type == "all":
            print("exposure :", cam.get(cv2.CAP_PROP_EXPOSURE))
            print("audo_exposure :", cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))
            print("brightness :", cam.get(cv2.CAP_PROP_BRIGHTNESS))
            print("saturation :", cam.get(cv2.CAP_PROP_SATURATION))
            print("contrast :", cam.get(cv2.CAP_PROP_CONTRAST))
            print("gain :", cam.get(cv2.CAP_PROP_GAIN))
            print("temperature :", cam.get(cv2.CAP_PROP_TEMPERATURE))
            print("fourcc :", cam.get(cv2.CAP_PROP_FOURCC))
            print("format :", cam.get(cv2.CAP_PROP_FORMAT))
            print("focus :", cam.get(cv2.CAP_PROP_FOCUS))
            print("mode :", cam.get(cv2.CAP_PROP_MODE))
            print("zoom :", cam.get(cv2.CAP_PROP_ZOOM))

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

class CameraDisplayer(Process):
    def __init__(self, cam_id, config, frame_queue, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"CameraDisplayer {self.cam_id}"
        self.screen_name = f"camera {cam_id}"
        self.config = config
        self.frame_queue = frame_queue

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
        while self.shared_dict[self.process_name]['running'] and self.frame_queue:
            if not self.frame_queue.empty():
                current_time = time.time()
                times.append(round(current_time - prev_time, 2))
                times = times[-maxCount:]
                status = self.shared_dict[self.process_name]
                status['fps'] = 1 / np.mean(times) #計算時間
                self.shared_dict[self.process_name] = status
                prev_time = current_time

                frame = self.frame_queue.get()
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
                time.sleep(0.005)


def asyn_open_cameras(cam_ids, config):
    camera_threads = {}
    display_threads = {}
    for cam_id in cam_ids:
        reader = CameraReader(cam_id, config)
        reader.start()
        camera_threads[cam_id] = reader
    for cam_id in cam_ids:
        displayer = CameraDisplayer(cam_id, config, camera_threads[cam_id].queue)
        displayer.start()
        display_threads[cam_id] = displayer
    return camera_threads, display_threads

def stop_all_threads(threads):
    for t in threads:
        t.running = False
    for t in threads:
        t.join()