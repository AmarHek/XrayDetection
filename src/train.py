import os
from ultralytics import YOLO

# Training parameters
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 10
SAVE = True
SAVE_PERIOD = 2
PROJECT = "vindr_yolov8"
name = "initial_run"

config_path = "../configs/"
data_config = os.path.join(config_path, "data.yaml")
hyp_config = os.path.join(config_path, "hyp.yaml")

# Load the model
model = YOLO("yolov8m.pt")

# Display the model information
model.info()

# Train the model
results = model.train(data=data_config,
                      epochs=EPOCHS,
                      batch=BATCH_SIZE,
                      imgsz=IMG_SIZE,
                      SAVE=SAVE,
                      project=PROJECT,
                      save_period=SAVE_PERIOD,
                      device="cuda",
                      hyp=hyp_config)
