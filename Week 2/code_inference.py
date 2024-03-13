# Import necessary libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import cv2, random

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import  DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# Import from utils file
from utils import register_kitti_mots_datasets

# Import matplotlib for visualization
import matplotlib.pyplot as plt

model = "Faster R-CNN"  # Set this to "Faster R-CNN" or "Mask R-CNN" to switch between models
assert model in ["Faster R-CNN", "Mask R-CNN", "Retinanet"]

if model == "Faster R-CNN":
    model_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" # change 50 to 101 for mask_rcnn_R_101_FPN_3x
    output_path = "./output/faster_rcnn"
elif model == "Mask R-CNN":
    model_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # change 50 to 101 for mask_rcnn_R_101_FPN_3x
    output_path = "./output/mask_rcnn"
elif model == "Retinanet":
    model_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml" # change 50 to 101 for mask_rcnn_R_101_FPN_3x
    output_path = "./output/retinanet"

#Setup configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_path))

# Set confidence threshold for this model
if model == "Retinanet":
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
else:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)

# Register dataset
register_kitti_mots_datasets(cfg)
# Setup COCO Evaluator for the test dataset
evaluator_test_faster_rcnn = COCOEvaluator("kitti_mots_test", output_dir=output_path)
test_loader_faster_rcnn = build_detection_test_loader(cfg, "kitti_mots_test")
# Run evaluation on the test dataset for Faster R-CNN
inference_on_dataset(DefaultPredictor(cfg).model, test_loader_faster_rcnn, evaluator_test_faster_rcnn)

# Visualize results for a few images
dataset_dicts_test = DatasetCatalog.get("kitti_mots_test")
kitti_mots_metadata_test = MetadataCatalog.get("kitti_mots_test")
for idx,d in enumerate(random.sample(dataset_dicts_test, 10)):
    img = cv2.imread(d["file_name"], cv2.IMREAD_COLOR)
    outputs = DefaultPredictor(cfg)(img)
    # Filter out instances that are not cars nor pedestrians
    instances = outputs["instances"].to("cpu")
    filtered_instances = instances[(instances.pred_classes == 2) | (instances.pred_classes == 0)] # Pedestrians and cars
    v = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata_test, scale=1.2)
    v = v.draw_instance_predictions(filtered_instances)
    plt.figure(figsize=(24, 20))
    plt.imshow(v.get_image())
    plt.title(f"{model} Predictions")
    plt.axis('off')
    plt.savefig(f'{output_path}/predictions_{idx}.png')
