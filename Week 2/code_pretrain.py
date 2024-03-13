# Import necessary libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import os, cv2, random

# Import from utils file
from utils import register_kitti_mots_datasets, LossEvalHook

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import  DatasetCatalog, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer

# Import matplotlib for visualization
import matplotlib.pyplot as plt

class CustomTrainer(DefaultTrainer):
    """
        Custom trainer to regularly produce Validation stats.
    """
    def __init__(self, cfg, val_period=100):
        super().__init__(cfg)
        self.val_period = val_period

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    # By @zepman at https://stackoverflow.com/questions/73293964/how-can-i-show-validation-loss-and-validation-accuracy-evaluation-for-the-detect
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.val_period, # Frequency of calculation
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

model = "Faster R-CNN"  # Set this to "Faster R-CNN" or "Mask R-CNN" to switch between models
assert model in ["Faster R-CNN", "Mask R-CNN"]

if model == "Faster R-CNN":
    model_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" # change 50 to 101 for mask_rcnn_R_101_FPN_3x
    output_path = "./output/finetuned_faster_rcnn"
else:
    model_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # change 50 to 101 for mask_rcnn_R_101_FPN_3x
    output_path = "./output/finetuned_mask_rcnn"

#Setup configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_path))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
register_kitti_mots_datasets(cfg, pretrain=True)
cfg.DATASETS.TRAIN = ("kitti_mots_train",)
cfg.DATASETS.TEST = ("kitti_mots_val",)
if model == "Mask R-CNN":
    cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 16  # "batch size" 
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.OUTPUT_DIR = output_path

finetune = False
if finetune:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(
        cfg, 
        val_period=cfg.TEST.EVAL_PERIOD # Compute loss on validation dataset
        )
    trainer.resume_or_load(resume=False)
    trainer.train()

# Load the saved weights (Pretrained model)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

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
    filtered_instances = instances[(instances.pred_classes == 1) | (instances.pred_classes == 0)] # Pedestrians and cars
    v = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata_test, scale=1.2)
    v = v.draw_instance_predictions(filtered_instances)
    plt.figure(figsize=(24, 20))
    plt.imshow(v.get_image())
    plt.title(f"{model} Predictions")
    plt.axis('off')
    plt.savefig(f'{output_path}/predictions_{idx}.png')

