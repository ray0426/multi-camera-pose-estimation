import os
import cv2
import sys
from multiprocessing import Process, Queue
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
    def __init__(self, cam_id, config, original_image, pose_2d, input_camera_name, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"PoseEstimator {self.cam_id}"
        self.config = config
        self.original_image = original_image
        self.pose_2d = pose_2d
        self.input_camera_name = input_camera_name
        self.shared_dict = shared_dict
        self.shared_dict[self.process_name] = {
            'fps': 0,
            'running': True,
            'poseKeypoints': None
        }

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
            # queue cancel_join_thread() to prevent block join_thread()
            # https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues
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
        prev_image_id = -1
        while not self.shared_dict["control signals"][self.process_name]["halt"]:
            input_frame = np.frombuffer(self.original_image.get_obj(), dtype=np.uint8).reshape((frameHeight, frameWidth, 3))
            if (input_frame is not None) and not (prev_image_id == self.shared_dict[self.input_camera_name]['image_id']):
                prev_image_id = self.shared_dict[self.input_camera_name]['image_id']
                current_time = time.time()
                times.append(round(current_time - prev_time, 2))
                times = times[-maxCount:]
                local_dict = self.shared_dict[self.process_name]
                local_dict['fps'] = 1 / np.mean(times) #計算時間
                prev_time = current_time
                self.shared_dict[self.process_name] = local_dict

                datum.cvInputData = input_frame    
                #opWrapper.emplaceAndPop([datum]) #原本的但有問題
                opWrapper.emplaceAndPop(global_op.VectorDatum([datum])) #改成這個程式碼
                # OutputData = datum.cvOutputData #加上骨架後影像
                poseKeypoints = datum.poseKeypoints
                faceKeypoints = datum.faceKeypoints
                handKeypoints = datum.handKeypoints
                pose_2d = np.frombuffer(self.pose_2d.get_obj(), dtype=np.float32).reshape((25, 3))
                if poseKeypoints is not None:
                    pose_2d[:] = poseKeypoints[0]
                else:
                    pass
                    # pose_2d[:] = pose_2d[:] * 0
                # 縮放 OutputData 成指定大小
                # resized_output = cv2.resize(OutputData, (frameWidth, frameHeight))
                # local_dict['poseKeypoints'] = poseKeypoints
            else:
                # tprint(f"queue {self.cam_id} empty!")
                # time.sleep(0.01)
                pass
            # tprint(f"                      queue size: {self.queue.qsize()}")
        print("finish hpe")

        opWrapper.stop()
        print("openpose closed")