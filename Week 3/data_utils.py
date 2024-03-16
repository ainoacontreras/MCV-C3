import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

# class MIT_split_dataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = sorted(os.listdir(root_dir))
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
#         self.images = self._load_images()

#     def _load_images(self):
#         images = []
#         for class_name in self.classes:
#             class_dir = os.path.join(self.root_dir, class_name)
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 images.append((img_path, self.class_to_idx[class_name]))
#         return images

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path, label = self.images[idx]
#         img = read_image(img_path).float()

#         if self.transform:
#             img = self.transform(img)

#         return img, label
    
# PairDataset will need to be defined to handle pairs creation for training
class PairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = np.array([label for _, label in dataset])
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}

    def __getitem__(self, index):
        target = np.random.randint(0, 2)  # Binary target: same class or not
        img1, label1 = self.dataset[index]
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            label2 = np.random.choice(list(set(self.labels) - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[label2])
        img2, _ = self.dataset[siamese_index]
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)
    
# Define the custom transform function
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