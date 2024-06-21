import detectron2
from detectron2.utils.logger import setup_logger
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
import cv2

setup_logger()

# Load an example image
im = cv2.imread("path/to/your/image.jpg")

# Set up configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.MODEL.DEVICE = "cuda"  # or "cpu"

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Visualize the results
v = Visualizer(im[:, :, ::-1], scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Detection Result", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
