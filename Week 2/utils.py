# Import some common libraries
import os, logging, time, datetime
import torch
import numpy as np

# Import Detectron2 utilities
from detectron2.data import  DatasetCatalog, MetadataCatalog
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from pycocotools.mask import toBbox
from detectron2.structures import BoxMode
from pathlib import Path


def get_kitti_mots_dicts(data_folders, pretrain=False):
    images_path= 'mcv/datasets/C5/KITTI-MOTS/training/image_02/'
    text_notations_folder= 'mcv/datasets/C5/KITTI-MOTS/instances_txt/'
    dataset = []
    image_id = 1  # Initialize image_id
    for folder in data_folders:
        text_file = os.path.join(text_notations_folder, f'{folder}.txt')
        images_path_full = os.path.join(images_path, folder)
        images_list = sorted(list(Path(images_path_full).glob('*.png')))

        with open(text_file, 'r') as f:
            old_frame = None
            objs = []
            record = {}
            
            for line in f:
                frame_index, ob_id, class_id, height, width, lre = list(map(int, line.strip().split(' ')[:5])) + [line.strip().split(' ')[5]]
                if class_id == 10 or class_id == 10000: continue  # Skip unwanted classes

                id_tracking = ob_id % 1000
                rle_object = {'size': [height, width], 'counts': lre.encode('utf-8')}
                bbox_coordinates = toBbox(rle_object).tolist()

                if pretrain:
                    category_id = class_id if class_id == 1 else 0
                else:
                    category_id = 2 if class_id == 1 else 0
                obj = {
                    "id": id_tracking,
                    "bbox": bbox_coordinates,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": rle_object,
                    "category_id": category_id
                }

                if frame_index != old_frame:  # New frame or first frame
                    if old_frame is not None:  # Not first frame, save previous frame data
                        record["annotations"] = objs
                        dataset.append(record)

                    # Start new record
                    record = {
                        "file_name": str(images_list[frame_index]),
                        "image_id": image_id,
                        "height": height,
                        "width": width,
                        "annotations": []  # Prepare for new objects
                    }
                    objs = [obj]  # Start with current object
                    old_frame = frame_index
                    image_id += 1
                else:  # Same frame, continue adding objects
                    objs.append(obj)

            # Add last record if exists
            if objs:
                record["annotations"] = objs
                dataset.append(record)

    return dataset


def register_kitti_mots_datasets(cfg, pretrain=False):
    test_seq_ids = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']
    train_seq_ids = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017']
    val_seq_ids = ['0019', '0020']
    classes = ["person", "car"] if pretrain else MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    # Register datasets
    DatasetCatalog.register("kitti_mots_train", lambda: get_kitti_mots_dicts(train_seq_ids, pretrain=pretrain))
    MetadataCatalog.get("kitti_mots_train").set(thing_classes=classes)

    DatasetCatalog.register("kitti_mots_val", lambda: get_kitti_mots_dicts(val_seq_ids, pretrain=pretrain))
    MetadataCatalog.get("kitti_mots_val").set(thing_classes=classes)

    DatasetCatalog.register("kitti_mots_test", lambda: get_kitti_mots_dicts(test_seq_ids, pretrain=pretrain))
    MetadataCatalog.get("kitti_mots_test").set(thing_classes=classes)


# From https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)