import os
from ultralytics import YOLO

PROJECT = "vindr_yolov8"
name = "initial_run"

# Training parameters
EPOCHS = 300
IMG_SIZE = 640
PATIENCE = 50
BATCH_SIZE = 16
SAVE = True

# lr scheduling parameters
LR0 = 0.001
LRF = 0.01
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# Loss parameters
BOX = 0.05
CLS = 0.5
IOU = 0.5

# Augmentation parameters
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 2.0
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPLR = 0.5
FLIPUP = 0.0
MOSAIC = 0.0
COPY_PASTE = 0.0
ERASE = 0.1


config_path = os.path.join(os.getcwd(), "configs")
data_config = os.path.join(config_path, "data.yaml")

# Load the model
model = YOLO("yolov8m.pt")

# Display the model information
model.info()

# Train the model
results = model.train(data=data_config, device="cuda",
                      project=PROJECT, name=name, plots=True,
                      epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, save=SAVE,
                      box=BOX, cls=CLS, iou=IOU, patience=PATIENCE, lr0=LR0, lrf=LRF,
                      hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
                      degrees=DEGREES, translate=TRANSLATE, scale=SCALE, shear=SHEAR,
                      perspective=PERSPECTIVE, fliplr=FLIPLR, flipud=FLIPUP,
                      mosaic=MOSAIC, copy_paste=COPY_PASTE, erase=ERASE)
