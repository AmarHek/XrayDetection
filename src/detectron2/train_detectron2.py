import os
import cv2
import random
import json
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Paths to your data
root = "/scratch/hekalo/Datasets/vindr/dataset/"
image_dir = os.path.join(root, "images")
annotation_dir = os.path.join(root, "annotations")
train_json = os.path.join(annotation_dir, "instances_train.json")
val_json = os.path.join(annotation_dir, "instances_val.json")
test_json = os.path.join(annotation_dir, "instances_test.json")
train_dir = os.path.join(image_dir, "train")
val_dir = os.path.join(image_dir, "val")
test_dir = os.path.join(image_dir, "test")

# Register your datasets
register_coco_instances("vindr_train", {}, train_json, train_dir)
register_coco_instances("vindr_val", {}, val_json, val_dir)
register_coco_instances("vindr_test", {}, test_json, test_dir)

# Verify the registration
my_dataset_train_metadata = MetadataCatalog.get("vinDr_train")
dataset_dicts = DatasetCatalog.get("vinDr_train")

# Visualize the dataset (optional)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('sample', vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Use pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
cfg.SOLVER.MAX_ITER = 1000    # Number of iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Number of classes in your dataset (update accordingly)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the model
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# Inference with the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)

v = Visualizer(image[:, :, ::-1],
               metadata=my_dataset_train_metadata,
               scale=0.5,
               instance_mode=ColorMode.SEGMENTATION)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('prediction', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
