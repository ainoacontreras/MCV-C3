import torch
import tqdm

def train(model, dataloader, optimizer, loss_func, miner, device):
    
    model.train()
    training_loss = 0.0
    for (images1, images2), targets in tqdm.tqdm(dataloader, total=len(dataloader)):
        
        images1, images2, targets = images1.to(device), images2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs1, outputs2 = model(images1, images2)
        targets = targets.float()  # Ensure targets are float for contrastive loss
        
        # Miner step to select informative pairs
        hard_pairs = miner(outputs1, outputs2, targets)
        
        # Computing loss on selected pairs
        loss = loss_func(outputs1, outputs2, targets, indices_tuple=hard_pairs)
        
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    
    training_loss /= len(dataloader.dataset)
    return training_loss

def validate(model, dataloader, loss_func, device):
    
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for (images1, images2), targets in tqdm.tqdm(dataloader, total=len(dataloader)):
            images1, images2, targets = images1.to(device), images2.to(device), targets.to(device)
            outputs1, outputs2 = model(images1, images2)
            targets = targets.float()  # Ensure targets are float for contrastive loss
            loss = loss_func(outputs1, outputs2, targets)
            validation_loss += loss.item()
    
    validation_loss /= len(dataloader.dataset)
    return validation_loss