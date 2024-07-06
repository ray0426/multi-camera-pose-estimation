import cv2
import numpy as np
import time
import threading
from singleton_lock import SingletonLock

print_lock = SingletonLock.get_lock('print')

def open_single_camera(camID, config):
    if config['resolution_fps_setting'] == "640x480@30":
        frameWidth  = 640
        frameHeight = 480
        fps = 30
    elif config['resolution_fps_setting'] == "1280x720@60":
        frameWidth  = 1280
        frameHeight = 720
        fps = 60
    elif config['resolution_fps_setting'] == "1920x1080@30":
        frameWidth  = 1920
        frameHeight = 1080
        fps = 30
    cap = cv2.VideoCapture(camID, config['api'])
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

def syn_open_multiple_cameras(camIDs, config):
    # 打開相機 (0 代表默認相機)
    if config['resolution_fps_setting'] == "640x480@30":
        frameWidth  = 640
        frameHeight = 480
        fps = 30
    elif config['resolution_fps_setting'] == "1280x720@60":
        frameWidth  = 1280
        frameHeight = 720
        fps = 60
    elif config['resolution_fps_setting'] == "1920x1080@30":
        frameWidth  = 1920
        frameHeight = 1080
        fps = 30
    # 打開相機
    cap0 = cv2.VideoCapture(camIDs[0], config['api'])
    cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap0.set(cv2.CAP_PROP_FPS, fps)
    cap0.set(cv2.CAP_PROP_EXPOSURE, config['exposure'])
    cap0.set(cv2.CAP_PROP_GAIN, config['gain'])
    cap1 = cv2.VideoCapture(camIDs[1], config['api'])
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

class camThread(threading.Thread):
    def __init__(self, previewName, camID, config):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.config = config
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID, self.config)

def print_cam_informations(camID, cam, type="simple"):
    with print_lock:
        print("=============" + "camera " + str(camID) + "=============")
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

def camPreview(previewName, camID, config):
    # 打開相機 (0 代表默認相機)
    if config['resolution_fps_setting'] == "640x480@30":
        frameWidth  = 640
        frameHeight = 480
        fps = 30
    elif config['resolution_fps_setting'] == "1280x720@60":
        frameWidth  = 1280
        frameHeight = 720
        fps = 60
    elif config['resolution_fps_setting'] == "1920x1080@30":
        frameWidth  = 1920
        frameHeight = 1080
        fps = 30

    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID, config['api'])
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cam.set(cv2.CAP_PROP_FPS, fps)
    cam.set(cv2.CAP_PROP_EXPOSURE, config['exposure'])
    cam.set(cv2.CAP_PROP_GAIN, config['gain'])

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
        prev_time = time.time()
    else:
        rval = False
    print_cam_informations(camID, cam)
    
    times = [1]
    maxCount = 20
    while rval:
        current_time = time.time()
        times.append(round(current_time - prev_time, 2))
        times = times[-maxCount:]
        optime = np.mean(times) #計算時間
        prev_time = current_time
        cv2.putText(frame,'FPS : {0:.2f}'.format(round(1/optime,2)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame,'Time : {0:.2f}'.format(round(optime, 5)), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        # 按下 'q' 鍵退出
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyWindow(previewName)

def asyn_open_cameras(camIDs, config):
    threads = []
    for camID in camIDs:
        threads.append(camThread("Camera " + str(camID), camID, config))
        threads[-1].start()