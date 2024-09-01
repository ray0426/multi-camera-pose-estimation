import cv2
import os

# Input video file path and output images save path
video_fn = "outputs/calibration/2024-09-01_10-30-06_1.avi"
images_path = "outputs/calibration/1"

# Ensure the output directory exists, create it if it doesn't
if not os.path.exists(images_path):
    os.makedirs(images_path)

# Open the video file
cap = cv2.VideoCapture(video_fn)

# Check if the video file was successfully opened
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_fn}")
    exit()

# Initialize frame counter
frame_count = 0

while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # Exit the loop if there are no more frames to read

    # Construct the output image filename
    img_filename = os.path.join(images_path, f"frame_{frame_count:05d}.jpg")

    # Save the image
    cv2.imwrite(img_filename, frame)

    # Print save information
    print(f"Saved {img_filename}")

    # Increment frame counter
    frame_count += 1

# Release the video capture object
cap.release()

print("Finished extracting frames.")