from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np

class CustomTransform:
    def __init__(self, config, mode):
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif mode == 'clip':
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.CenterCrop((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.ToTensor(),
            ])

    def __call__(self, img):
        return self.transform(img)
    
class CocoMetricDataset(Dataset):
    def __init__(self, root, captions_file, transforms):
        self.root = root
        self.transforms = transforms
        self.labels = []
        self.filenames = []
        self.captions = []

        #get 5000 random indices from the captions file
        random.seed(123)
        indices = random.sample(range(len(captions_file["annotations"])), 5000)
        
        for caption_info in tqdm(np.array(captions_file["annotations"])[indices], desc="Creating image-answer pairs..."):
            caption = caption_info["caption"]
            image_id = caption_info["image_id"]

            for img_info in captions_file["images"]:
                if img_info["id"] == image_id:
                    self.labels.append(image_id)
                    self.filenames.append(img_info["file_name"])
                    self.captions.append(caption)
                    break
        
    def __getitem__(self, index):
        # Load image and transform it
        image_filename = self.filenames[index]
        image = Image.open(os.path.join(self.root, image_filename)).convert('RGB')
        image = self.transforms(image)

        # Get label and caption
        label = self.labels[index] 
        caption = self.captions[index]

        return image, caption, label
    
    def __len__(self):
        return len(self.filenames)
    
def coco_collator(batch):
    """
    Format COCO data to ResNet50/FastText input format.
    """
    images, captions, labels = [], [], []
    for (img, caption, label) in batch:
        images.append(img)
        captions.append(caption)
        labels.append(label)
    return torch.stack(images), captions, torch.tensor(labels)
    
class CocoMetricTrainingDataset(Dataset):
    """
        Create COCO data to be used for training a triplet network.
    """
    def __init__(self, root, captions_file, id2img_dict, transforms):
        self.root = root
        self.transforms = transforms
        
        self.captions = []
        self.image_ids = []
        self.filenames = []

        for caption_info in tqdm(captions_file["annotations"], desc="Creating image-answer pairs..."):
            caption = caption_info["caption"]
            image_id = caption_info["image_id"]

            self.image_ids.append(image_id)
            self.filenames.append(id2img_dict[image_id])
            self.captions.append(caption)

        
    def __getitem__(self, index):
        # Load image and transform it
        image_filename = self.filenames[index]
        image = Image.open(os.path.join(self.root, image_filename)).convert('RGB')
        image = self.transforms(image)

        # Get label and caption
        caption = self.captions[index]

        return image, caption
    
    def __len__(self):
        return len(self.filenames)
    
def coco_metric_collator(batch):
    """
    Format COCO data to the triplet network training format
    """
    images, captions = [], []
    for (img, caption) in batch:
        images.append(img)
        captions.append(caption)
    return torch.stack(images), captions