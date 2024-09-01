import cv2
import numpy as np
import glob
import json

def extract_camera_parameters(
        # 兩個相機的資料夾
        camera0_image_dir='outputs/calibration/0/*.jpg', 
        camera1_image_dir='outputs/calibration/1/*.jpg',
        # 輸出json檔名
        output_dir='outputs/calibration/parameters.json',
        # 設定棋盤格大小 (寬度, 高度)
        checkerboard_size = (8, 6),
        # (mm) 實際每個方格的大小，例如1.0 cm
        square_size = 28
    ):
    # 準備棋盤格點在世界坐標系中的位置，例如 (0,0,0), (1,0,0), (2,0,0), ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存儲棋盤格點和圖像點
    objpoints = []  # 在世界坐標系中的3D點
    imgpoints_0 = []  # 在左視角圖像中的2D點
    imgpoints_1 = []  # 在右視角圖像中的2D點

    # 加載左右相機的圖像
    images_0 = sorted(glob.glob(camera0_image_dir))
    images_1 = sorted(glob.glob(camera1_image_dir))

    for img_0, img_1 in zip(images_0, images_1):
        img0 = cv2.imread(img_0)
        img1 = cv2.imread(img_1)
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # 尋找左右相機的棋盤格角點
        ret0, corners0 = cv2.findChessboardCorners(gray0, checkerboard_size, None)
        ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)

        if ret0 and ret1:
            objpoints.append(objp)
            imgpoints_0.append(corners0)
            imgpoints_1.append(corners1)
            # cv2.drawChessboardCorners(img0, (8,6), corners0, ret0)
            # cv2.imshow('img', img0)
            # cv2.waitKey(1500)
            # cv2.drawChessboardCorners(img1, (8,6), corners1, ret1)
            # cv2.imshow('img', img1)
            # cv2.waitKey(1500)

    # 校準兩個相機
    ret0, mtx0, dist0, _, _ = cv2.calibrateCamera(objpoints, imgpoints_0, gray0.shape[::-1], None, None)
    ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_1, gray1.shape[::-1], None, None)

    # 進行立體校準
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret, mtx0, dist0, mtx1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_0, imgpoints_1, mtx0, dist0, mtx1, dist1, gray0.shape[::-1], criteria=criteria, flags=flags)
    
    parameters = {
        'ret': ret,
        'mtx0': mtx0.tolist(),
        'dist0': dist0.tolist(),
        'mtx1': mtx1.tolist(),
        'dist1': dist1.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist()
    }
    with open(output_dir, 'w') as fout:
        json.dump(parameters, fout)
        

    return ret, mtx0, dist0, mtx1, dist1, R, T, E, F
    # print()
    # print(ret)
    # print('左相機矩陣: ')
    # print(mtxL)
    # print('左相機畸形: ')
    # print(distL)
    # print('右相機矩陣: ')
    # print(mtxR)
    # print('右相機畸形: ')
    # print(distR)
    # print('旋轉矩陣: ')
    # print(R)
    # print('平移向量: ')
    # print(T)
    # print('本質矩陣E: ')
    # print(E)
    # print('基礎矩陣F: ')
    # print(F)
    # print()

'''
distortion coefficient
k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [tx ty]]]]
'''
if __name__ == '__main__':
    ret, mtx0, dist0, mtx1, dist1, R, T, E, F = extract_camera_parameters()