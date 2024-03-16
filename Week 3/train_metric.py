from pytorch_metric_learning import miners, losses, distances
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader, random_split
from data_utils import *
from train_utils import *
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import wandb
wandb.login(key='14a56ed86de5bf43e377d95d05458ca8f15f5017')

config = {
    'IMG_WIDTH': 256,
    'IMG_HEIGHT': 256,
    'TRAINING_DATASET_DIR': '../Week 1/data/MIT_split/train',
    'TEST_DATASET_DIR': '../Week 1/data/MIT_split/test',
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'n_neighbors': 5,
    'type_model': 'siamese',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

wandb.init(project='week3', entity='c5-g8', config=config)

transform_train = CustomTransform(config, mode='train')
transform_test = CustomTransform(config, mode='test')

train_dataset = datasets.ImageFolder(root=config['TRAINING_DATASET_DIR'], transform=transform_train)
test_dataset =  datasets.ImageFolder(root=config['TEST_DATASET_DIR'], transform=transform_test)

total_length = len(train_dataset)
train_size = int(0.8 * total_length)  # e.g., 80% for training
valid_size = total_length - train_size  # remaining 20% for validation

# Split dataset
train_dataset, validation_dataset = random_split(train_dataset, [train_size, valid_size])

dataloader_train = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
dataloader_validation = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Model Definition remains the same
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

model = Net()
model.to(config['device'])

distance = distances.LpDistance(power=2)

if config['type_model'] == 'siamese':
    loss_func = losses.ContrastiveLoss(pos_margin=0.2, neg_margin=0.8, distance=distance)
    miner = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)

else:
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance)
    miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

accuracy_calculator = AccuracyCalculator(include=("precision_at_1", 'mean_average_precision'), k=config['n_neighbors'])

best_acc = 0
# Adjusted Training Loop with Miner
for epoch in range(config['epochs']):

    train_loss = train(model, dataloader_train, optimizer, loss_func, miner, config['device'], config['type_model'])
    accuracies = test(model, train_dataset, validation_dataset, accuracy_calculator)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Acc: {accuracies['precision_at_1']}, MAP: {accuracies['mean_average_precision']}")

    wandb.log({'Train Loss': train_loss, 'Validation Acc': accuracies['precision_at_1'], 'MAP': accuracies['mean_average_precision']})

    if accuracies['precision_at_1'] > best_acc:
        best_acc = accuracies['precision_at_1']
        torch.save(model.state_dict(), 'pretrained/best_model_siamese.pth')

print(f"Best Validation Accuracy: {best_acc}")
print("Testing the model...")
test_accuracies = test(model, train_dataset, test_dataset, accuracy_calculator)
print(f"Test Accuracy: {test_accuracies['precision_at_1']}, MAP: {test_accuracies['mean_average_precision']}")
