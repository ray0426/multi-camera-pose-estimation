import queue
import threading
import tkinter as tk
from tkinter import messagebox
from read_camera import CameraReader, CameraDisplayer

from singleton_lock import tprint

class CameraControlPanel(tk.Frame):
    def __init__(self, master=None, config=None, camera_ids=[0, 1]):
        super().__init__(master)
        self.master = master
        self.config = config
        self.camera_ids = camera_ids
        self.camera_threads = {}
        self.display_threads = {}
        self.read_fps_labels = {}
        self.display_fps_labels = {}
        self.create_widgets()
        self.update_fps()

        # Bind the delete window event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        for cam_id in self.camera_ids:
            cam_panel_frame = tk.Frame(self)
            cam_panel_frame.pack(pady=10)

            # Camera ID
            id_frame = tk.Frame(cam_panel_frame)
            id_frame.pack()
            info_label = tk.Label(id_frame, text=f"Camera {cam_id}")
            info_label.pack()

            # Two FPS info
            fps_frame = tk.Frame(cam_panel_frame)
            fps_frame.pack()
            read_fps_label = tk.Label(fps_frame, text="Read FPS: 0")
            read_fps_label.pack(side="left")
            self.read_fps_labels[cam_id] = read_fps_label

            display_fps_label = tk.Label(fps_frame, text=" Display FPS: 0")
            display_fps_label.pack(side="left")
            self.display_fps_labels[cam_id] = display_fps_label
            
            # buttons
            button_frame = tk.Frame(cam_panel_frame)
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
        
    def start_camera(self, cam_id):
        if cam_id not in self.camera_threads.keys():
            reader = CameraReader(cam_id, self.config)
            reader.start()
            self.camera_threads[cam_id] = reader
            tprint(f"Camera {cam_id} started!")

    def stop_camera(self, cam_id):
        if cam_id in self.camera_threads.keys():
            self.camera_threads[cam_id].running = False
            self.camera_threads[cam_id].join()
            del self.camera_threads[cam_id]
            tprint(f"Camera {cam_id} stopped!")
    
    def start_display(self, cam_id):
        if cam_id in self.camera_threads.keys() and \
            cam_id not in self.display_threads.keys():
            displayer = CameraDisplayer(
                cam_id, self.config, 
                self.camera_threads[cam_id].queue
            )
            displayer.start()
            self.display_threads[cam_id] = displayer
            tprint(f"Display {cam_id} started!")

    def stop_display(self, cam_id):
        if cam_id in self.display_threads.keys():
            self.display_threads[cam_id].running = False
            self.display_threads[cam_id].join()
            del self.display_threads[cam_id]
            tprint(f"Display {cam_id} stopped!")

    def update_fps(self):
        for cam_id in self.camera_ids:
            if cam_id in self.camera_threads.keys():
                self.read_fps_labels[cam_id].config(
                    text = f"Read FPS: {self.camera_threads[cam_id].fps:.2f}"
                )
            else:
                self.read_fps_labels[cam_id].config(
                    text = f"Read FPS: invalid"
                )
        for cam_id in self.camera_ids:
            if cam_id in self.display_threads.keys():
                self.display_fps_labels[cam_id].config(
                    text = f" Display FPS: {self.display_threads[cam_id].fps:.2f}"
                )
            else:
                self.display_fps_labels[cam_id].config(
                    text = f" Display FPS: invalid"
                )
        self.after(1000, self.update_fps)

    def on_closing(self):
        # Close every thread before closing the panel
        for cam_id in list(self.camera_threads.keys()):
            self.stop_camera(cam_id)
        for cam_id in list(self.display_threads.keys()):
            self.stop_display(cam_id)
        self.master.destroy()
