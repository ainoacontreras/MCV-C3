import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 7)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.fc1 = nn.Linear(21632, 2048)
        self.fc2 = nn.Linear(2048, 768)
        self.fc3 = nn.Linear(768, num_classes)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x