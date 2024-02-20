import torch
import numpy as np
import torch.nn.functional as F
import tqdm

def train(model, data_loader, criterion, optimizer, device):

    model.train()
    training_loss = 0.0
    correct = 0.0
    for i, (data, target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data).to("cpu")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        probs = F.softmax(output.detach(), dim=1)
        pred_labels = np.argmax(probs, axis=1)
        correct += (pred_labels == target).sum()

        training_loss += loss.item()
    
    training_loss /= len(data_loader.dataset)
    training_accuracy = 100. * correct / len(data_loader.dataset)
    return training_loss, training_accuracy

def validate(model, data_loader, criterion, device):

    model.eval()
    validation_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, (data, target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            data = data.to(device)
            output = model(data).to("cpu")
            loss = criterion(output, target)
            validation_loss += loss
            probs = F.softmax(output, dim=1)
            pred_labels = np.argmax(probs, axis=1)
            correct += (pred_labels == target).sum()

    validation_loss /= len(data_loader.dataset)
    validation_accuracy = 100. * correct / len(data_loader.dataset)
    return validation_loss, validation_accuracy