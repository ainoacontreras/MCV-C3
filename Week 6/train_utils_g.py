import torch
import torch.nn as nn
from tqdm.auto import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
    
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (image, audio, text, y) in tqdm(enumerate(dataloader), desc="Validating...", total=len(dataloader)):
            # Send data to target device
            image, audio, y = image.to(device), audio.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(image, audio, text)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc  


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (image, audio, text, y) in tqdm(enumerate(dataloader), desc="Training...", total=len(dataloader)):

        # Send data to target device
        image, audio, y = image.to(device), audio.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(image, audio, text)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          early_stop_thresh: int = 5,
          checkpoints_folder: str = None,
          model_name: str = None):
    
    # 2. Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_accuracy = -1
    best_epoch = -1
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.6f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {test_loss:.6f} | "
            f"valid_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if (test_acc > best_accuracy):
            best_accuracy = test_acc
            best_epoch = epoch
            # save model checkpoint
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_fn,
                        },
                        f'{checkpoints_folder}/{model_name}_checkpoint.pt')

            # print("model saved")
                    
        # if did not improve in the last "early_stop_thresh" epochs, reduce learning rate
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stop at epoch %d" % epoch)
            break  # terminate the training loop            

    # 6. Return the filled results at the end of the epochs
    return results