import os
import cv2
import sys
import json
from multiprocessing import Process
import time
import numpy as np
from singleton_lock import print_lock, tprint
from utils import decode_frame_size_rate
from utils import BODY25_SKELETON_EDGES

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseEstimator3D(Process):
    def __init__(self, cam_id, config, pose_2d, camera_ids, shared_dict):
        super().__init__()
        self.cam_id = cam_id
        self.process_name = f"PoseEstimator3D {self.cam_id}"
        self.config = config
        self.pose_2d = pose_2d
        self.camera_ids = camera_ids
        self.shared_dict = shared_dict
        self.shared_dict[self.process_name] = {
            'fps': 0,
            'running': True
        }

    def run(self):
        tprint(f"Start 3D HPE {self.cam_id}")
        try:
            self.pose_estimation_3D()
        except Exception as e:
            tprint(f"Exception in 3D HPE: {e}")
        finally:
            tprint(f"Stopping 3D HPE")
    
    def pose_estimation_3D(self):
        frameWidth, frameHeight, fps = decode_frame_size_rate(self.config['resolution_fps_setting'])

        # read camera parameters
        with open('outputs/calibration/parameters.json', 'r') as fin:
            parameters = json.load(fin)

        # 3D plot initialization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 設置軸的範圍
        ax.set_xlim([350, 550])
        ax.set_ylim([-175, -110])
        ax.set_zlim([-400, 150])

        # 初始化24條線條，初始時設置為空
        lines = [ax.plot([], [], [], 'bo-', linewidth=2)[0] for _ in range(24)]

        prev_time = time.time()
        times = [1]
        maxCount = 60
        prev_image_id = -1
        while not self.shared_dict["control signals"][self.process_name]["halt"]:
            current_time = time.time()
            times.append(round(current_time - prev_time, 2))
            times = times[-maxCount:]
            local_dict = self.shared_dict[self.process_name]
            local_dict['fps'] = 1 / np.mean(times) #計算時間
            prev_time = current_time
            self.shared_dict[self.process_name] = local_dict

            pose_2ds = {}
            for cam_id in self.camera_ids:
                pose_2ds[cam_id] = np.copy(np.frombuffer(self.pose_2d[cam_id].get_obj(), dtype=np.float32).reshape((25, 3)))

            # process each keypoint
            pose_3d = np.zeros((25, 4), dtype=np.float32)
            for i in range(25):
                u0, v0, c0 = pose_2ds[0][i]
                u1, v1, c1 = pose_2ds[1][i]
                if c0 < 0.1 or c1 < 0.1:
                    continue
                vec3 = self.camera_2D_to_global(parameters, [[u0, v0], [u1, v1]])
                M = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ])
                vec3_rotate = M @ vec3
                pose_3d[i, 0:3] = vec3_rotate.T
                pose_3d[i, 3] = c0 * c1

            for i in range(len(BODY25_SKELETON_EDGES)):
                start_idx = BODY25_SKELETON_EDGES[i, 0]
                end_idx = BODY25_SKELETON_EDGES[i, 1]
                x1, y1, z1, c1 = pose_3d[start_idx]
                x2, y2, z2, c2 = pose_3d[end_idx]
                if c1 > 0.15 and c2 > 0.15:
                    # 隨機生成兩個3D點
                    point1 = [x1, y1, z1]
                    point2 = [x2, y2, z2]
                    
                    # 更新線條的數據並顯示
                    lines[i].set_data([point1[0], point2[0]], [point1[1], point2[1]])
                    lines[i].set_3d_properties([point1[2], point2[2]])
                    lines[i].set_visible(True)
                else:
                    # 隱藏不需要顯示的線條
                    lines[i].set_visible(False)
            
            # 刷新視窗
            plt.draw()
            plt.pause(0.01)  # 暫停0.01秒以便更新視窗

            # time.sleep(1)
        tprint("finish 3D  hpe")
    
    def camera_2D_to_global(self, parameters, vec):
        u0 = vec[0][0]
        v0 = vec[0][1]
        u1 = vec[1][0]
        v1 = vec[1][1]

        mtx0 = np.array(parameters['mtx0'])
        mtx1 = np.array(parameters['mtx1'])
        dist0 = np.array(parameters['dist0'])
        dist1 = np.array(parameters['dist1'])
        R = np.array(parameters['R'])
        T = np.array(parameters['T'])
        
        distorted_point_0 = np.array([[[u0, v0]]], dtype=np.float32)
        undistorted_point_0 = cv2.undistortPoints(distorted_point_0, mtx0, dist0, P=mtx0)
        distorted_point_1 = np.array([[[u1, v1]]], dtype=np.float32)
        undistorted_point_1 = cv2.undistortPoints(distorted_point_1, mtx1, dist1, P=mtx1)

        direction_0 = undistorted_point_0[0][0]
        direction_1 = undistorted_point_1[0][0]

        ray_0_camera = np.array([[direction_0[0]], [direction_0[1]], [1.0]])
        ray_1_camera = np.array([[direction_1[0]], [direction_1[1]], [1.0]])

        ray_0_world = ray_0_camera
        ray_1_world = np.linalg.inv(R) @ ray_1_camera  # 旋轉
        ray_1_world += T                              # 平移

        # start of 0: (0, 0, 0)
        # direction of 0: ray_0_world
        # start of 1: T
        # direction of 1: ray_1_world
        O_L = np.array([[0], [0], [0]])  # 左相機起點 (世界坐標系)
        O_R = T  # 右相機起點 (世界坐標系中的T)
        d_L = ray_0_world / np.linalg.norm(ray_0_world)  # 左相機的方向向量，歸一化
        d_R = ray_1_world / np.linalg.norm(ray_1_world)  # 右相機的方向向量，歸一化

        # 計算 t 和 s
        A = d_L.T @ d_L
        B = d_L.T @ d_R
        C = d_R.T @ d_R
        D = (O_R - O_L).T @ d_L
        E = (O_R - O_L).T @ d_R

        denom = A * C - B * B

        if denom != 0:
            t = (B * E - C * D) / denom
            s = (A * E - B * D) / denom
        else:
            t, s = 0, 0  # 射線平行的情況，使用某個默認值

        # 射線上的最近點
        P_L_closest = O_L + t * d_L
        P_R_closest = O_R + s * d_R

        # 計算兩個最近點的中點作為最終的3D位置
        P_3D = (P_L_closest + P_R_closest) / 2

        return P_3D

