import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
from datetime import datetime
from torch.nn import functional as f
import numpy as np
from matplotlib import pyplot as plt

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
    
config = {  'IMG_WIDTH': 32, 
            'IMG_HEIGHT': 32, 
            'BATCH_SIZE': 50,
            'NUMBER_OF_EPOCHS': 50,
            'TRAINING_DATASET_DIR': './data/MIT_small_train_1/train/',
            'TEST_DATASET_DIR': './data/MIT_split/test/',
            'VALIDATION_DATASET_DIR': './data/MIT_small_train_1/test/',
            'layers_to_train': 177, # list containing the modules to train 
            'learning_rate': 0.001,
            'momentum': 0.0,
            'nesterov': False,
            'optimizer': 'Adam', 
            'scheduler': None, 
            'MODEL_FNAME': '14', 
            'MODE': 'train'
            }

transform = transforms.Compose([
    transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #change from rgb to bgr
    transforms.Lambda(lambda x: x/255.0),
])

dataset_train = MIT_split_dataset(config['TRAINING_DATASET_DIR'], transform=transform)
dataset_validation = MIT_split_dataset(config['VALIDATION_DATASET_DIR'], transform=transform)
dataset_test = MIT_split_dataset(config['TEST_DATASET_DIR'], transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=50, shuffle=True)
dataloader_validation = torch.utils.data.DataLoader(dataset_train, batch_size=50, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=807, shuffle=False)

class CNN(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) 
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2) 
        self.batchnorm32 = torch.nn.BatchNorm2d(32)
        self.dropout = torch.nn.Dropout(0.4)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x) # [50, 32, 32, 32]
        x = self.relu(x)
        x = self.maxpool(x) 
        x = self.batchnorm32(x) # [50, 32, 16, 16]
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x) 
        x = self.batchnorm32(x) # [50, 32, 8, 8]
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x) # [50, 32, 4, 4]
        x = self.batchnorm32(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        #x = self.softmax(x)

        return x


assert torch.cuda.is_available(), "GPU is not enabled"
# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN()
model.to(device)
torch.manual_seed(0) # seed for reproductibility
criterion = torch.nn.CrossEntropyLoss() # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) #betas atr include momentum=0.9

def train(epochs, criterion, model, optimizer, train_loader, val_loader):
    epoch_number = 0

    best_vloss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # training one epoch
        training_loss = 0.0
        correct = 0.0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            data = data.to(device)
            output = model(data).to("cpu")
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            output = f.softmax(output.detach(), dim=1)
            pred = np.argmax(output, axis=1)
            correct += (pred==target).sum()

            training_loss += loss.item() 
        training_loss /= len(train_loader.dataset)
        training_accuracy = 100. * correct / len(train_loader.dataset)

        validation_loss = 0.0
        correct = 0.0
        model.eval()

        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(val_loader):
                vinputs = vinputs.to(device)
                voutputs = model(vinputs).to("cpu")
                vloss = criterion(voutputs, vlabels)
                validation_loss += vloss
                voutputs = f.softmax(voutputs, dim=1)
                pred = np.argmax(voutputs, axis=1)
                #pred = voutputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += (pred==vlabels).sum()

        validation_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        print('TRAIN loss {} acc {} VAL loss {} acc {}'.format(training_loss, training_accuracy, validation_loss, accuracy))

        # Track best performance, and save the model's state
        if validation_loss < best_vloss:
            best_vloss = validation_loss
            model_path = 'model_1.pth'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
    return model

model = train(config['NUMBER_OF_EPOCHS'], criterion, model, optimizer, dataloader_train, dataloader_validation)

correct = 0.0
test_loss = 0.0
model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader_test):
        inputs = inputs.to(device)
        outputs = model(inputs).to("cpu")
        loss = criterion(outputs, labels)
        test_loss += loss
        outputs = f.softmax(outputs, dim=1)
        pred = np.argmax(outputs, axis=1)
        correct = (pred==labels).sum()
        #pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        #correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(dataloader_test.dataset)
    accuracy = 100. * correct / len(dataloader_test.dataset)
    print('TEST loss {} acc {}'.format(test_loss, accuracy))