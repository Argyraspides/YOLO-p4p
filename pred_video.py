import os
import cv2
from ultralytics import YOLO
from os import putenv
import numpy as np
from PIL import Image

# Set the environment variable for AMD GPUs
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Load the best model path
with open('exportPath.txt', 'r') as f:
    bestModelPath = f.read().strip()

modelPath = os.path.join(os.getcwd(), bestModelPath.replace(".onnx", ".pt"))
model = YOLO(modelPath)

# Set paths for input video and output video
inputVideoPath = os.path.join(os.getcwd(), 'escooters/dataBank', 'FOOTAGE', '20240524T140450.mp4')
outputVideoPath = os.path.join(os.getcwd(), 'escooters', 'predictions_video', 'output_video.mp4')

# Open the video file
cap = cv2.VideoCapture(inputVideoPath)
if not cap.isOpened():
    print(f"Error opening video file: {inputVideoPath}")
    exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(outputVideoPath, fourcc, fps, (frame_width, frame_height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Get the annotated frame (use plot method)
    annotated_frame = results[0].plot()

    # Convert the annotated frame to a format that OpenCV can handle
    if isinstance(annotated_frame, Image.Image):
        annotated_frame = np.array(annotated_frame)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release video capture and writer objects
cap.release()
out.release()

print(f"Processed video saved at {outputVideoPath}")
