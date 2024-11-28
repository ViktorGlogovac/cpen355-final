import os
from roboflow import Roboflow
from ultralytics import YOLO
import cv2

# Step 1: Initialize Roboflow and download the dataset
rf = Roboflow(api_key="upR0TpbLVIlvsGJwoHe5")
project = rf.workspace("cpen-355").project("score-keep-ml")
version = project.version(1)

# Download the dataset in yolov8 format
dataset = version.download("yolov8")

# Step 2: Initialize the YOLO model

# Initialize the YOLO model with a pre-trained model
model = YOLO('yolov8m.pt')

# Construct the path to the 'data.yaml' file from the downloaded dataset
data_yaml_path = os.path.join(dataset.location, 'data.yaml')

# Ensure that the dataset is correctly downloaded and 'data.yaml' exists
if os.path.exists(data_yaml_path):
    print(f"Training using data from: {data_yaml_path}")
    
    # Train the YOLO model with adjusted parameters
    model.train(
        data=data_yaml_path,   # Path to data.yaml file
        epochs=300,            # Increased number of epochs
        imgsz=640,             # Image size
        batch=16,              # Batch size
        name='score-keep1',
        val=True,              # Include validation
        workers=4,             # Number of workers for data loading
        optimizer='Adam',      # Use a different optimizer
        lr0=1e-3,              # Adjusted initial learning rate
        patience=10            # Early stopping after 10 epochs with no improvement
    )
else:
    print("Error: 'data.yaml' file not found. Please check the dataset download path.")