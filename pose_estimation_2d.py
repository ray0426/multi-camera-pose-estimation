import os
import cv2
import sys
import threading
from multiprocessing import Process, Queue
import queue
import time
import numpy as np
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate

global_op = None
OPENPOSE_VALID = False

def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    else:
        return gettrace() is not None

def load_openpose_module():
    global global_op, OPENPOSE_VALID
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
        global_op = op
        OPENPOSE_VALID = True
    except Exception as e:
        print(e)
        return False
    return True

class PoseEstimator(Process):
# class PoseEstimator(threading.Thread):
    def __init__(self, cam_id, config, frame_queue):
        super().__init__()
        # threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.config = config
        self.frame_queue = frame_queue
        self.queue = Queue(maxsize = 5)
        # self.queue = queue.Queue(maxsize = 5)
        self.fps = 0
        self.running = True

    def run(self):
        if not OPENPOSE_VALID:
            print("Loading OpenPose Start")
            if not load_openpose_module():
                print("OpenPose invalie!")
                return
            print("Loading OpenPose Success")
            
            
        tprint(f"Start HPE {self.cam_id}")
        try:
            self.pose_estimation()
        except Exception as e:
            tprint(f"Exception in HPE {self.cam_id}: {e}")
        finally:
            tprint(f"Stopping HPE {self.cam_id}")
    
    def pose_estimation(self):
        frameWidth, frameHeight, fps = decode_frame_size_rate(self.config['resolution_fps_setting'])
        OPENPOSE_PATH = "D:/coding/pose_estimation_projects/openpose/"

        #------------選擇模式--------------
        params = dict()
        params["model_folder"] = OPENPOSE_PATH + "/models/"
        params["net_resolution"] = '128x192'
        params["model_pose"] = "BODY_25"
        params["render_pose"] = 1
        params["face"] = False
        params["hand"] = False
        #---------------------------------

        #----------引入openpose--------
        opWrapper = global_op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        datum = global_op.Datum()
        #-----------------------------

        prev_time = time.time()
        times = [1]
        maxCount = 60
        while self.running and self.frame_queue:
            if not self.frame_queue.empty():
                current_time = time.time()
                times.append(round(current_time - prev_time, 2))
                times = times[-maxCount:]
                self.fps = 1 / np.mean(times) #計算時間
                prev_time = current_time

                frame = self.frame_queue.get()
                datum.cvInputData = frame    
                #opWrapper.emplaceAndPop([datum]) #原本的但有問題
                opWrapper.emplaceAndPop(global_op.VectorDatum([datum])) #改成這個程式碼
                OutputData = datum.cvOutputData #加上骨架後影像
                # 縮放 OutputData 成指定大小
                resized_output = cv2.resize(OutputData, (frameWidth, frameHeight))
                if self.queue.full():
                    self.queue.get()
                self.queue.put(resized_output)
            else:
                # tprint(f"queue {self.cam_id} empty!")
                time.sleep(0.01)
            # tprint(f"                      queue size: {self.queue.qsize()}")