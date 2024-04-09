from tqdm import tqdm
import torch
import numpy as np

def train(loader, model, optimizer, criterion, device):
    """
        Training Loop for the triplet network approach.
    """
    model.train()
    total_loss, loss_list = 0, []
    for (images, captions) in tqdm(loader, total=len(loader), desc="Training..."):
        optimizer.zero_grad()
        images = images.to(device)
        visual_embeddings, textual_embeddings = model(images, captions)
    
        loss = criterion(visual_embeddings, textual_embeddings)
        if type(loss) in (tuple, list):
            loss = loss[0]
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())

    return total_loss, np.mean(loss_list)

def test(loader, model, criterion, device):
    model.eval()
    total_loss, loss_list = 0, []
    with torch.no_grad():
        for (images, captions) in tqdm(loader, total=len(loader), desc="Testing..."):  
            images = images.to(device)
            visual_embeddings, textual_embeddings = model(images, captions)
            loss = criterion(visual_embeddings, textual_embeddings)
            if type(loss) in (tuple, list):
                loss = loss[0]
            total_loss += loss.item()
            loss_list.append(loss.item())
    return total_loss, np.mean(loss_list)