from tqdm import tqdm
import torch
import numpy as np

def train(loader, model, optimizer, criterion, device):
    """
        Training Loop for the triplet network approach.
    """
    model.train()
    total_loss, loss_list = 0, []
    total_generated_negs = 0
    for (images, captions) in tqdm(loader, total=len(loader), desc="Training..."):
        optimizer.zero_grad()
        images = images.to(device)
        visual_embeddings, textual_embeddings = model(images, captions)
    
        loss, generated_negs = criterion(visual_embeddings, textual_embeddings, captions)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())

        total_generated_negs += generated_negs

    return total_loss, np.mean(loss_list), total_generated_negs/len(loader.dataset)

def test(loader, model, criterion, device):
    """
        Inference loop.
    """
    model.eval()
    total_loss, loss_list = 0, []
    with torch.no_grad():
        for (images, captions) in tqdm(loader, total=len(loader), desc="Testing..."):  
            images = images.to(device)
            visual_embeddings, textual_embeddings = model(images, captions)
            loss, _ = criterion(visual_embeddings, textual_embeddings)
            if type(loss) in (tuple, list):
                loss = loss[0]
            total_loss += loss.item()
            loss_list.append(loss.item())
    return total_loss, np.mean(loss_list)