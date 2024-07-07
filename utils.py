

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