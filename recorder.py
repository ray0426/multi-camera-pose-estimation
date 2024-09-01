import os
import cv2
import time
import numpy as np
import datetime
from multiprocessing import Process
from singleton_lock import tprint
from utils import decode_frame_size_rate

class Recorder(Process):
    def __init__(self, config, original_images, shared_dict, save_path = "./outputs"):
        super().__init__()
        self.config = config
        self.original_images = original_images
        self.shared_dict = shared_dict
        self.save_path = save_path
        self.process_name = "Recorder"
    
    def run(self):
        tprint(f"Start recorder")
        try:
            self.record()
        except Exception as e:
            tprint(f"Exception in Recorder: {e}")
        finally:
            for cam_id, writer in self.video_writers.items():
                writer.release()
            tprint(f"Stopping Recorder")

    def record(self):
        frameWidth, frameHeight, fps = decode_frame_size_rate(self.config['resolution_fps_setting'])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_fns = {}
        self.video_writers = {}
        prev_image_ids = {}
        for cam_id, _ in self.original_images.items():
            output_fns[cam_id] = os.path.join(self.save_path, f"{now}_{cam_id}.avi")
            self.video_writers[cam_id] = cv2.VideoWriter(output_fns[cam_id],
                fourcc,
                int(fps),
                (frameWidth, frameHeight))
            prev_image_ids[cam_id] = -1

        while not self.shared_dict["control signals"][self.process_name]["halt"]:
            for cam_id, frame_array in self.original_images.items():
                if not (prev_image_ids[cam_id] == self.shared_dict[f"CameraReader {cam_id}"]['image_id']):
                    prev_image_ids[cam_id] = self.shared_dict[f"CameraReader {cam_id}"]['image_id']

                    frame = np.frombuffer(frame_array.get_obj(), dtype=np.uint8).reshape((frameHeight, frameWidth, 3))
                    self.video_writers[cam_id].write(frame)
                    # print(f"write frame {prev_image_ids[cam_id]} of camera {cam_id}")

photo_id = 0

def photo(self):
    global photo_id
    frameWidth, frameHeight, fps = decode_frame_size_rate(self.config['resolution_fps_setting'])
    output_dir = "outputs/calibration"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for cam_id in self.camera_ids:
        # image_id = self.shared_dict[f"CameraReader {cam_id}"]['image_id']
        output_fn = os.path.join(output_dir, f"{cam_id}", f"frame_{photo_id:05d}.jpg")
        frame = np.frombuffer(self.original_image[cam_id].get_obj(), dtype=np.uint8).reshape((frameHeight, frameWidth, 3))
        save_image(frame, output_fn)
    
    print(f"Saved")
    photo_id = photo_id + 1

def save_image(frame, output_fn):
    # Save the image
    cv2.imwrite(output_fn, frame)
    
    # Optionally, print the save information
    print(f"Saved {output_fn}")