import os
import cv2
import random
import json
import torch

from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine.hooks import EvalHook, PeriodicWriter

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

# parameters
model_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
batch_size = 8
num_workers = 2
lr = 0.00025
max_iter = 50000
batch_size_per_image = 128
num_classes = 14

# Register your datasets
register_coco_instances("vindr_train", {}, train_json, train_dir)
register_coco_instances("vindr_val", {}, val_json, val_dir)
register_coco_instances("vindr_test", {}, test_json, test_dir)

# Verify the registration
vindr_train_metadata = MetadataCatalog.get("vindr_train")
dataset_dicts = DatasetCatalog.get("vindr_train")


# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_config))
cfg.DATASETS.TRAIN = ("vindr_train",)
cfg.DATASETS.TEST = ("vindr_val",)
cfg.TEST.EVAL_PERIOD = 1500
cfg.DATALOADER.NUM_WORKERS = num_workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)  # Use pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = lr  # Learning rate
cfg.SOLVER.MAX_ITER = max_iter    # Number of iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Number of classes in your dataset (update accordingly)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# Custom Hook to save the best model based on validation loss
class BestCheckpointerHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.best_loss = float("inf")
        self.best_model_weights = None

    def after_step(self):
        # Ensure evaluation only happens at the end of each epoch
        if self.trainer.iter % self.cfg.TEST.EVAL_PERIOD == 0 and self.trainer.iter != 0:
            val_loss = self._compute_validation_loss()
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_weights = self.trainer.model.state_dict()
                torch.save(self.best_model_weights, os.path.join(self.cfg.OUTPUT_DIR, "best_model.pth"))

    def _compute_validation_loss(self):
        self.trainer.model.eval()
        val_loss = 0
        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        with torch.no_grad():
            for batch in val_loader:
                loss_dict = self.trainer.model(batch)
                losses = sum(loss_dict.values())
                val_loss += losses.item()
        self.trainer.model.train()
        return val_loss / len(val_loader)


# Custom Trainer to add BestCheckpointerHook
class BestModelTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointerHook(self.cfg))
        hooks = [hook for hook in hooks if not isinstance(hook, EvalHook)]
        hooks.append(EvalHook(self.cfg.TEST.EVAL_PERIOD, self.test))
        return hooks


# Train the model
trainer = BestModelTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Load the best model for inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
predictor = DefaultPredictor(cfg)

# Evaluate the model on the test set
evaluator = COCOEvaluator("vindr_test", cfg, False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "vindr_test")
print(inference_on_dataset(predictor.model, test_loader, evaluator))

