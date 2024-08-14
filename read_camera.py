import cv2
import numpy as np
import time
from process_base_class import ProcessBaseClass
from multiprocessing import Queue
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