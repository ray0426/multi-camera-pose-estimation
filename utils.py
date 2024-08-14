import cv2

def decode_frame_size_rate(setting_str):
    if setting_str == "640x480@30":
        frameWidth  = 640
        frameHeight = 480
        fps = 30
    elif setting_str == "1280x720@60":
        frameWidth  = 1280
        frameHeight = 720
        fps = 60
    elif setting_str == "1920x1080@30":
        frameWidth  = 1920
        frameHeight = 1080
        fps = 30
    return frameWidth, frameHeight, fps

def print_cam_informations(cam_id, cam, type="simple"):
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