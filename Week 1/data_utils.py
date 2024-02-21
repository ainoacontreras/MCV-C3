import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

class MIT_split_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = read_image(img_path).float()

        if self.transform:
            img = self.transform(img)

        return img, label
    
# Define the custom transform function
class CustomTransform:
    def __init__(self, config, mode):

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.Lambda(lambda x: x/255.0),
                transforms.RandomAffine(
                    degrees=20,  # rotation_range
                    translate=(0.2, 0.2),  # width_shift_range, height_shift_range
                    scale=(0.8, 1.2),  # zoom_range, inverse because PyTorch scales with 1/value
                    shear=20,  # shear_range
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),  # horizontal_flip
                transforms.ToPILImage(),
                transforms.RandomApply([transforms.ColorJitter(brightness=(0.8, 1.2))], p=1.0),  # brightness_range
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
                transforms.Lambda(lambda x: x/255.0)
            ])

    def __call__(self, img):
        return self.transform(img)