from ultralytics import YOLO
import os

data_config = "/scratch/hekalo/Datasets/vindr/dataset/yolo_config.yaml"

# Load the model
model = YOLO("yolov8l.pt")

# Load the dataset
data = model.load_data(data_config)

# Check the number of images and labels loaded
print("Training images:", len(data['train']))
print("Validation images:", len(data['val']))
