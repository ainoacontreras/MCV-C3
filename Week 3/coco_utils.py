import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from collections import defaultdict
from PIL import Image
import random
import json

class COCOforTripletNetwork(torch.nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()
        self.backbone = fasterrcnn_resnet50_fpn(weights="COCO_V1").backbone
        self.linear = torch.nn.Linear(4096, embed_size)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x["pool"].flatten(1)
        x = self.linear(x)
        return x
    

def create_image_category_pairs(annotations, save_path="image_category_pairs.json"):
    """
    Given COCO annotations, assign to each sample a single object category.
    Save assignments into specified path.
    """
    images_dict = defaultdict(list)
    for k, v in annotations.items():
        for image_id in v:
            images_dict[image_id].append(int(k))

    final_dicts = []
    for image_id, objects in images_dict.items():
        sample = {}
        sample["image_id"] = image_id
        sample["all_categories"] = objects
        sample["selected_category"] = random.choice(objects)
        final_dicts.append(sample)

    with open(save_path, "w") as f:
        json.dump(final_dicts, f)

class COCODataset(Dataset):
    """
        Create dataset from metric learning with pairs of Image-object category from the
        COCO 2014 dataset.
    """
    def __init__(self, root, image_folder, instances_info, transforms, annotations=None, image_category_pairs=None, is_queries=False):
        self.root = root
        self.image_folder = image_folder
        self.transforms = transforms
        self.targets = []
        self.filenames = []
        self.all_image_cats = []
        
        # If not creating a dataset with the retrieval queries
        if not is_queries:
            for sample in image_category_pairs:
                # Find current image filename
                filename = None
                for item in instances_info["images"]:
                    if item['id'] == sample["image_id"]:
                        filename = item["file_name"]
                        break

                # Create a dataset instance with each image/category pair
                self.filenames.append(filename)
                self.targets.append(sample["selected_category"])
                self.all_image_cats.append(sample["all_categories"])
        else:
            # For all the images, find the associated objects
            images_dict = defaultdict(list)
            for k, v in annotations.items():
                for image_id in v:
                    images_dict[image_id].append(int(k))

            for image_id, objects in images_dict.items():
                # Find current image filename
                for item in instances_info["images"]:
                    if item['id'] == image_id:
                        filename = item["file_name"]
                        break
                # For the queries, we keep all the object categories
                self.filenames.append(filename)
                self.targets.append(objects)

        

    # obtain the sample with the given index
    def __getitem__(self, index):
        # Load image and transform it
        image_filename = self.filenames[index]
        image = Image.open(os.path.join(self.root, self.image_folder, image_filename)).convert('RGB')
        image = self.transforms(image)
        # Get label of current image
        target = self.targets[index]
        return image, target
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)