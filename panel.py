import os
import time
import numpy as np
import tkinter as tk
from multiprocessing import Manager, Array
from camera_reader import CameraReader
from camera_displayer import CameraDisplayer
from pose_estimation_2d import PoseEstimator
from pose_estimation_3d import PoseEstimator3D
from recorder import Recorder, photo
import ctypes

from singleton_lock import tprint

PROCTYPES = ["CameraReader", "CameraDisplayer", "PoseEstimator"]

class CameraControlPanel(tk.Frame):
    def __init__(self, master=None, config=None, camera_ids=[0, 1]):
        super().__init__(master)
        self.master = master
        self.config = config
        self.camera_ids = camera_ids

        self.processes = {
            "CameraReader": {},
            "CameraDisplayer": {},
            "PoseEstimator": {},
            "Recorder": {},
            "PoseEstimator3D": {}
        }

        self.read_fps_labels = {}
        self.display_fps_labels = {}
        self.hpe_fps_labels = {}

        self.process_manager = Manager()
        self.shared_dict = self.process_manager.dict()
        self.shared_dict["control signals"] = {}
        self.original_image = {}
        self.pose_2d = {}

        self.create_widgets()
        self.update_fps()

        # Bind the delete window event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        for cam_id in self.camera_ids:
            cam_panel_frame = tk.Frame(self)
            cam_panel_frame.pack(pady=10)

            input_panel_frame = tk.Frame(cam_panel_frame)
            input_panel_frame.pack(padx=10, side="left")

            # Camera ID
            id_frame = tk.Frame(input_panel_frame)
            id_frame.pack()
            info_label = tk.Label(id_frame, text=f"Camera {cam_id}")
            info_label.pack()

            # Two FPS info
            fps_frame = tk.Frame(input_panel_frame)
            fps_frame.pack()
            read_fps_label = tk.Label(fps_frame, text="Read FPS: 0")
            read_fps_label.pack(side="left")
            self.read_fps_labels[cam_id] = read_fps_label

            display_fps_label = tk.Label(fps_frame, text=" Display FPS: 0")
            display_fps_label.pack(side="left")
            self.display_fps_labels[cam_id] = display_fps_label
            
            # buttons
            button_frame = tk.Frame(input_panel_frame)
            button_frame.pack()
            start_read_button = tk.Button(
                button_frame, 
                text = "Start Camera", 
                command = lambda cid = cam_id: self.start_camera(cid)
            )
            start_read_button.pack(side = "left")

            stop_read_button = tk.Button(
                button_frame, 
                text = "Stop Camera", 
                command = lambda cid = cam_id: self.stop_camera(cid)
            )
            stop_read_button.pack(side = "left")

            start_display_button = tk.Button(
                button_frame, 
                text = "Start Display", 
                command = lambda cid = cam_id: self.start_display(cid)
            )
            start_display_button.pack(side = "left")

            stop_display_button = tk.Button(
                button_frame, 
                text = "Stop Display", 
                command = lambda cid = cam_id: self.stop_display(cid)
            )
            stop_display_button.pack(side = "left")

            # Human Pose Estimation (HPE) panel
            hpe_panel_frame = tk.Frame(cam_panel_frame)
            hpe_panel_frame.pack(padx=10, side="left")

            # Camera ID (hidden)
            id_frame = tk.Frame(hpe_panel_frame)
            id_frame.pack()
            info_label = tk.Label(id_frame, text=f"")
            info_label.pack()

            # HPE FPS info
            fps_frame = tk.Frame(hpe_panel_frame)
            fps_frame.pack()
            hpe_fps_label = tk.Label(fps_frame, text="HPE FPS: 0")
            hpe_fps_label.pack(side="left")
            self.hpe_fps_labels[cam_id] = hpe_fps_label
            
            # buttons
            button_frame = tk.Frame(hpe_panel_frame)
            button_frame.pack()
            start_hpe_button = tk.Button(
                button_frame, 
                text = "Start HPE", 
                command = lambda cid = cam_id: self.start_hpe(cid)
            )
            start_hpe_button.pack(side = "left")

            stop_hpe_button = tk.Button(
                button_frame, 
                text = "Stop HPE", 
                command = lambda cid = cam_id: self.stop_hpe(cid)
            )
            stop_hpe_button.pack(side = "left")
        
        # record
        record_panel_frame = tk.Frame(self)
        record_panel_frame.pack(pady=10)
        button_frame = tk.Frame(record_panel_frame)
        button_frame.pack()
        start_record_button = tk.Button(
            button_frame, 
            text = "Start Record", 
            command = self.start_record
        )
        start_record_button.pack(side = "left")
        stop_record_button = tk.Button(
            button_frame, 
            text = "Stop Record", 
            command = self.stop_record
        )
        stop_record_button.pack(side = "left")
        photo_button = tk.Button(
            button_frame, 
            text = "Photo", 
            command = self.take_photo
        )
        photo_button.pack(side = "left")

        # 3D HPE
        hpe_3D_panel_frame = tk.Frame(self)
        hpe_3D_panel_frame.pack(pady=10)
        button_frame = tk.Frame(hpe_3D_panel_frame)
        button_frame.pack()
        start_hpe_3D_button = tk.Button(
            button_frame, 
            text = "Start 3D HPE", 
            command = self.start_hpe_3D
        )
        start_hpe_3D_button.pack(side = "left")
        stop_hpe_3D_button = tk.Button(
            button_frame, 
            text = "Stop 3D HPE", 
            command = self.stop_hpe_3D
        )
        stop_hpe_3D_button.pack(side = "left")

    def start_process(self, process_class, cam_id, proc_type):
        if cam_id not in self.processes[proc_type].keys():
            if proc_type == "CameraReader":
                self.original_image[cam_id] = Array(ctypes.c_uint8, 720 * 1280 * 3) # magic number should be changed
                self.pose_2d[cam_id] = Array(ctypes.c_float, 25 * 3) # magic number should be changed
                process = process_class(
                    cam_id, self.config, 
                    self.original_image[cam_id], 
                    self.shared_dict
                )
            elif proc_type == "CameraDisplayer":
                process = process_class(
                    cam_id, self.config, 
                    self.original_image[cam_id],
                    self.pose_2d[cam_id],
                    f"CameraReader {cam_id}",
                    f"PoseEstimator {cam_id}",
                    self.shared_dict
                )
            elif proc_type == "PoseEstimator":
                process = process_class(
                    cam_id, self.config, 
                    self.original_image[cam_id],
                    self.pose_2d[cam_id],
                    f"CameraReader {cam_id}",
                    self.shared_dict
                )
            elif proc_type == "PoseEstimator3D":
                process = process_class(
                    cam_id, self.config,
                    self.pose_2d,
                    self.camera_ids,
                    self.shared_dict
                )
            local_control_dict = self.shared_dict["control signals"]
            local_control_dict[f"{proc_type} {cam_id}"] = {}
            local_control_dict[f"{proc_type} {cam_id}"]["halt"] = False
            self.shared_dict["control signals"] = local_control_dict
            process.start()
            self.processes[proc_type][cam_id] = process
            tprint(f"{proc_type} {cam_id} started!")

    def stop_process(self, cam_id, proc_type):
        if cam_id in self.processes[proc_type].keys():
            local_control_dict = self.shared_dict["control signals"]
            local_control_dict[f"{proc_type} {cam_id}"]["halt"] = True
            self.shared_dict["control signals"] = local_control_dict
            self.processes[proc_type][cam_id].join()
            del self.processes[proc_type][cam_id]
            if proc_type == "CameraReader":
                del self.original_image[cam_id]
            tprint(f"{proc_type} {cam_id} stopped!")
        
    def start_camera(self, cam_id):
        self.start_process(CameraReader, cam_id, 'CameraReader')

    def stop_camera(self, cam_id):
        self.stop_process(cam_id, 'CameraReader')
    
    def start_display(self, cam_id):
        self.start_process(CameraDisplayer, cam_id, 'CameraDisplayer')

    def stop_display(self, cam_id):
        self.stop_process(cam_id, 'CameraDisplayer')
    
    def start_hpe(self, cam_id):
        self.start_process(PoseEstimator, cam_id, 'PoseEstimator')

    def stop_hpe(self, cam_id):
        self.stop_process(cam_id, 'PoseEstimator')
    
    def start_hpe_3D(self):
        self.start_process(PoseEstimator3D, 0, 'PoseEstimator3D')

    def stop_hpe_3D(self):
        self.stop_process(0, 'PoseEstimator3D')

    def update_fps(self):
        for cam_id in self.camera_ids:
            for proc_type, labels in [('CameraReader', self.read_fps_labels), ('CameraDisplayer', self.display_fps_labels), ('PoseEstimator', self.hpe_fps_labels)]:
                if cam_id in self.processes[proc_type]:
                    status = self.shared_dict[f"{proc_type} {cam_id}"]
                    labels[cam_id].config(text=f"{proc_type.split('er')[0]} FPS: {status['fps']:.2f}")
                else:
                    labels[cam_id].config(text=f"{proc_type.split('er')[0]} FPS: invalid")
        self.after(1000, self.update_fps)

    def on_closing(self):
        # Close every proc before closing the panel
        for proc_type, proc_dict in self.processes.items():
            for cam_id in list(proc_dict.keys()):
                if proc_type == 'CameraReader':
                    self.stop_camera(cam_id)
                elif proc_type == 'CameraDisplayer':
                    self.stop_display(cam_id)
                elif proc_type == 'PoseEstimator':
                    self.stop_hpe(cam_id)
                elif proc_type == 'Recorder':
                    self.stop_record()
        time.sleep(1)
        self.master.destroy()

    def start_record(self):
        # check the environment is ready
        for cam_id in self.camera_ids:
            for proc_type in ["CameraReader"]:
                if cam_id not in self.processes[proc_type].keys():
                    tprint(f"{proc_type} {cam_id} is not ready")
                    return
        process = Recorder(
            self.config, 
            self.original_image,
            self.shared_dict
        )
        local_control_dict = self.shared_dict["control signals"]
        local_control_dict["Recorder"] = {}
        local_control_dict["Recorder"]["halt"] = False
        self.shared_dict["control signals"] = local_control_dict
        process.start()
        self.processes["Recorder"][0] = process
        tprint(f"Recorder started!")

    def stop_record(self):
        if 0 in self.processes["Recorder"].keys():
            local_control_dict = self.shared_dict["control signals"]
            local_control_dict["Recorder"]["halt"] = True
            self.shared_dict["control signals"] = local_control_dict
            self.processes["Recorder"][0].join()
            del self.processes["Recorder"][0]
            tprint(f"Recorder stopped!")

    # TODO: this might not be in panel process
    def take_photo(self):
        try:
            while True:
                photo(self)
                time.sleep(5)
        except Exception as e:
            print(f"Exception in take_photo : {e}")