from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
from tqdm import tqdm

class CustomTransform:
    def __init__(self, config, mode):
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        
        for caption_info in tqdm(captions_file["annotations"][:len(captions_file["annotations"])//8], desc="Creating image-answer pairs..."):
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
    return torch.stack(images), captions, torch.Tensor(labels).int()