import os
from ultralytics import YOLO

PROJECT = "vindr_yolov8"
name = "initial_run"

# Training parameters
EPOCHS = 500
IMG_SIZE = 640
PATIENCE = 50
BATCH_SIZE = 16
SAVE = True

# lr scheduling parameters
LR0 = 0.001
LRF = 0.001
WARMUP_EPOCHS = 5.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# Loss parameters
BOX = 5.0
CLS = 0.5
IOU = 0.7

# Augmentation parameters
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 5.0
TRANSLATE = 0.1
SCALE = 0.0
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPLR = 0.5
FLIPUP = 0.0
MOSAIC = 0.0
COPY_PASTE = 0.0
ERASING = 0.0
CROP_FRACTION = 1.0

config_path = os.path.join(os.getcwd(), "configs")
data_config = os.path.join(config_path, "data.yaml")

# Load the model
# model = YOLO("yolov8m.pt")
# model = YOLO("yolov8n.pt")
model = YOLO("yolov8l.pt")

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
                      mosaic=MOSAIC, copy_paste=COPY_PASTE, erasing=ERASING,
                      crop_fraction=CROP_FRACTION)
