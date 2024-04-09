import torch
from torch.utils.data import DataLoader, random_split
import json
import wandb
from model import Net
from mining_utils import HardMiner, TripletMarginLossWithMiner
import argparse
from train_utils import train, test
from data_utils import CustomTransform, CocoMetricTrainingDataset, coco_metric_collator

# WandB config
WANDB_PROJECT = None
WANDB_ENTITY = None
# wandb.login(key='')

# Parse commanda line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="image2text", choices=["image2text", "text2image"], help='Retrieval mode.')
parser.add_argument('--text_encoder', type=str, default="ft", choices=["ft", "bert"], help='Text encoding model.')
args = parser.parse_args()

config = {
    'IMG_WIDTH': 224,
    'IMG_HEIGHT': 224,
    'TRAINING_DATASET_DIR': "/home/mcv/datasets/C5/COCO/train2014",
    'TEST_DATASET_DIR': '/home/mcv/datasets/C5/COCO/val2014',
    'num_workers': 4,
    'batch_size': 32,
    'epochs': 15,
    'learning_rate': 1e-4,
    'n_neighbors': 5,
    'mode': args.mode,
    'margin': 0.5,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'text_encoder_type': args.text_encoder
}
   
# File containing images info (file_name)
with open(f"/home/mcv/datasets/C5/COCO/captions_train2014.json", "r") as f:
    captions_train = json.load(f)

# Relationship between image_id and its filename
id2img_dict = {i["id"]: i["file_name"] for i in captions_train["images"]}

# Dataset for finetuning the triplet network
finetuning_dataset = CocoMetricTrainingDataset(
    root="/home/mcv/datasets/C5/COCO/train2014", 
    captions_file=captions_train,
    id2img_dict=id2img_dict,
    transforms=CustomTransform({"IMG_WIDTH": 224, "IMG_HEIGHT": 224}, mode="train"))

# Create train/val/test splits
torch.manual_seed(0)
total_length = len(finetuning_dataset)
train_size = int(0.6 * total_length)  # e.g., 60% for training
valid_size = int(0.2 * total_length)  # e.g., 20% for validation
test_size = total_length - train_size - valid_size # remaining 20% for testing
train_dataset, validation_dataset, test_dataset = random_split(finetuning_dataset, [train_size, valid_size, test_size])

# Create Dataloaders
dataloader_train = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config["num_workers"], collate_fn=coco_metric_collator)
dataloader_validation = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config["num_workers"], collate_fn=coco_metric_collator)
dataloader_test = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config["num_workers"], collate_fn=coco_metric_collator)

# Initialize model
model = Net(config['text_encoder_type'])
model.to(config["device"])

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# Mining strategy
miner = HardMiner(config["margin"], mode=config["mode"])
criterion = TripletMarginLossWithMiner(config["margin"], miner, mode=config["mode"])

# Training Loop
run_name = f"{config["mode"]}_{config["mining_strat"]}_{config["text_encoder_type"]}_{config["learning_rate"]}"
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name, config=config)

best_loss = 0
for epoch in range(config['epochs']):
    train_loss, mean_train_loss = train(dataloader_train, model, optimizer, criterion, config['device'])
    val_loss, mean_val_loss = test(dataloader_validation, model, criterion, config['device'])
    
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Train Loss: {train_loss}")

    wandb.log({'Train Loss': train_loss, 'Val Loss': val_loss, 'Mean Train Loss': mean_train_loss, 'Mean Val Loss': mean_val_loss})

    if val_loss > best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'./pretrained/{run_name}.pth')

print(f"Best Validation Loss: {best_loss}")
# Load the best model found during training and test it
print("Testing the model...")
model.load_state_dict(torch.load(f'./pretrained/{run_name}.pth', map_location=config["device"]))
test_loss, mean_test_loss = test(dataloader_test, model, criterion, config['device'])
print(f"Test Loss: {test_loss}")