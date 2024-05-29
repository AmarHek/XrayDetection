import os
from ultralytics import YOLO

PROJECT = "vindr_yolov8"
name = "initial_run"

# Training parameters
EPOCHS = 100
IMG_SIZE = 640
PATIENCE = 10
BATCH_SIZE = 8
SAVE = True
LR0 = 0.01
LRF = 0.01

# Loss parameters
#BOX = 0.5
#CLS = 0.5
#IOU = 0.2

# Augmentation parameters
HSV_H = 0
HSV_S = 0
HSV_V = 0
DEGREES = 10
TRANSLATE = 0.1
SCALE = 0.05
FLIPLR = 0.5
FLIPUP = 0
ERASING = 0.05
CROP_FRACTION = 0.9


config_path = os.path.join(os.getcwd(), "configs")
data_config = os.path.join(config_path, "data.yaml")

# Load the model
model = YOLO("yolov8m.pt")

# Display the model information
model.info()

# Train the model
results = model.train(data=data_config,
                      device="cuda",
                      project=PROJECT, name=name, plots=True,
                      epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, save=SAVE,
                      # box=BOX, cls=CLS, iou=IOU,
                      patience=PATIENCE, lr0=LR0, lrf=LRF,
                      hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
                      degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
                      fliplr=FLIPLR, flipud=FLIPUP)
