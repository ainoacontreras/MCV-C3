import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data_utils import MIT_split_dataset, CustomTransform
from train_utils import train, validate
from model_luis import Model
import numpy as np
import wandb
wandb.login(key='14a56ed86de5bf43e377d95d05458ca8f15f5017')

config = {
    'IMG_WIDTH': 256,
    'IMG_HEIGHT': 256,
    'TRAINING_DATASET_DIR': 'data/MIT_small_train_1/train',
    'VALIDATION_DATASET_DIR': 'data/MIT_small_train_1/test',
    'TEST_DATASET_DIR': 'data/MIT_split/test',
    'learning_rate': 0.001,
    'batch_size': 32,
    'decay': 0.001,
    'n_epochs': 50
}

torch.manual_seed(123) # seed for reproductibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

transform_cifar = transforms.Compose([
    transforms.Resize((config['IMG_WIDTH'], config['IMG_HEIGHT'])), # Resize images to match your model's expected input size
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Lambda(lambda x: x/255.0) # Normalize image
])

# Load CIFAR-10 training and validation datasets
trainset_cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
valset_cifar = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

# Create dataloaders for CIFAR-10
dataloader_train_cifar = DataLoader(trainset_cifar, batch_size=config['batch_size'], shuffle=True)
dataloader_val_cifar = DataLoader(valset_cifar, batch_size=config['batch_size'], shuffle=False)

model = Model(num_classes=10).to(device)

# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

# best_val_loss = np.inf
# for epoch in range(config['n_epochs']//2):
#     train_loss, train_acc = train(model, dataloader_train_cifar, criterion, optimizer, device)
#     val_loss, val_acc = validate(model, dataloader_val_cifar, criterion, device)

#     print(f'Epoch {epoch+1}/{config["n_epochs"]}')
#     print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
#     print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'pretrained/best_model_cifar.pth')


# ----------------- MIT_split_dataset -----------------

wandb.init(project="week1", entity='c5-g8', config=config)

transform_train = CustomTransform(config, mode='train')
transform_test = CustomTransform(config, mode='test')

dataset_train = MIT_split_dataset(config['TRAINING_DATASET_DIR'], transform=transform_train)
dataset_validation = MIT_split_dataset(config['VALIDATION_DATASET_DIR'], transform=transform_test)
dataset_test = MIT_split_dataset(config['TEST_DATASET_DIR'], transform=transform_test)

dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=config['batch_size'], shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False)

state_dict = torch.load('pretrained/best_model_cifar.pth')

model = Model(num_classes=8).to(device)

for k, v in state_dict.items():
    if k.startswith('final_conv'):
        continue
    model.state_dict()[k] = v

# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

best_val_loss = np.inf
for epoch in range(config['n_epochs']):
    train_loss, train_acc = train(model, dataloader_train, criterion, optimizer, device)
    val_loss, val_acc = validate(model, dataloader_validation, criterion, device)

    print(f'Epoch {epoch+1}/{config["n_epochs"]}')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'pretrained/best_model.pth')

wandb.finish()
# Evaluate the model on the test data
test_loss, test_acc = validate(model, dataloader_test, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
