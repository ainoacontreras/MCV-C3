from pytorch_metric_learning import miners, losses, distances
import torch
from torch.utils.data import DataLoader, random_split
from data_utils import CustomTransform
from train_utils import train, test
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import os
from coco_utils import COCOforTripletNetwork, COCODataset
import wandb
import json
wandb.login(key='')

config = {
    'IMG_WIDTH': 256,
    'IMG_HEIGHT': 256,
    'ROOT_DIRECTORY': '/export/home/mcv/datasets/C5/COCO',
    'IMAGE_DIRECTORY': 'train2014',
    'batch_size': 64,
    'num_workers': 2,
    'epochs': 50,
    'learning_rate': 0.00001,
    'n_neighbors': 5,
    'type_model': 'triplet',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}
wandb.init(project='week3_coco', entity='c5-g8', name="coco_triplet_network", config=config)

# Initialize transforms
transform_train = CustomTransform(config, mode='train')
transform_test = CustomTransform(config, mode='test')

# File containing images information (file_name)
with open(os.path.join(config["ROOT_DIRECTORY"], "instances_train2014.json"), "r") as f:
    instances_info = json.load(f)

# COCO contains images with  multiple categories associated. We randomly select one category to 
# represent the whole image. This way, we avoid different category samples to have the same embeddings
with open("image_category_pairs.json", "r") as f:
    image_category_pairs = json.load(f)

# Create dataset
coco_dataset = COCODataset(
    root=config['ROOT_DIRECTORY'],
    image_folder=config["IMAGE_DIRECTORY"], 
    instances_info=instances_info, 
    transforms=transform_train,
    image_category_pairs=image_category_pairs)

# Split dataset into Train/Val/Test
total_length = len(coco_dataset)
train_size = int(0.6 * total_length)  # e.g., 60% for training
valid_size = int(0.2 * total_length)  # e.g., 20% for validation
test_size = total_length - train_size - valid_size # remaining 20% for testing
train_dataset, validation_dataset, test_dataset = random_split(coco_dataset, [train_size, valid_size, test_size])

dataloader_train = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config["num_workers"])
dataloader_validation = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config["num_workers"])
dataloader_test = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config["num_workers"])

# Initialize model
model = COCOforTripletNetwork()
model.to(config['device'])

# Triplet Loss/Mining approach
distance = distances.LpDistance(power=2)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance)
miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all")

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
accuracy_calculator = AccuracyCalculator(include=("precision_at_1", 'mean_average_precision'), k=config['n_neighbors'])

# Adjusted Training Loop with Miner
best_acc = 0
for epoch in range(config['epochs']):
    train_loss = train(model, dataloader_train, optimizer, loss_func, miner, config['device'], config['type_model'])
    accuracies = test(model, train_dataset, validation_dataset, accuracy_calculator)
    
    print(f"Epoch {epoch+1}, Validation Acc: {accuracies['precision_at_1']}, MAP: {accuracies['mean_average_precision']}, Train Loss: {train_loss}")

    wandb.log({'Train Loss': train_loss, 'Validation Acc': accuracies['precision_at_1'], 'MAP': accuracies['mean_average_precision']})

    if accuracies['precision_at_1'] > best_acc:
        best_acc = accuracies['precision_at_1']
        torch.save(model.state_dict(), './pretrained/best_model_cocotriplet_final_relu_smallerLR.pth')

print(f"Best Validation Accuracy: {best_acc}")
print("Testing the model...")
test_accuracies = test(model, train_dataset, test_dataset, accuracy_calculator)
print(f"Test Accuracy: {test_accuracies['precision_at_1']}, MAP: {test_accuracies['mean_average_precision']}")
