import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_utils import MIT_split_dataset, CustomTransform
from train_utils import train, validate
from model_luis import Model
import numpy as np

config = {
    'IMG_WIDTH': 256,
    'IMG_HEIGHT': 256,
    'TRAINING_DATASET_DIR': '../data/MIT_small_train_1/train',
    'VALIDATION_DATASET_DIR': '../data/MIT_small_train_1/validation',
    'TEST_DATASET_DIR': '../data/MIT_split/test',
    'learning_rate': 0.001,
    'batch_size': 64,
    'decay': 0.001,
    'n_epochs': 50
}

transform_train = CustomTransform(config, mode='train')
transform_test = CustomTransform(config, mode='test')

dataset_train = MIT_split_dataset(config['TRAINING_DATASET_DIR'], transform=transform_train)
dataset_validation = MIT_split_dataset(config['VALIDATION_DATASET_DIR'], transform=transform_test)
dataset_test = MIT_split_dataset(config['TEST_DATASET_DIR'], transform=transform_test)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
dataloader_validation = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(num_classes=8).to(device)
torch.manual_seed(123) # seed for reproductibility
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

for epoch in range(config['n_epochs']):
    train_loss, train_acc = train(model, dataloader_train, optimizer, criterion, device)
    val_loss, val_acc = validate(model, dataloader_validation, criterion, device)

    print(f'Epoch {epoch+1}/{config["n_epochs"]}')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Evaluate the model on the test data
test_loss, test_acc = validate(model, dataloader_test, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")