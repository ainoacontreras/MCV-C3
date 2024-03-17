import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
    
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