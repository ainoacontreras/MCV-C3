import torch
import tqdm
from pytorch_metric_learning import testers

def train(model, dataloader, optimizer, loss_func, miner, device, type_model):
    
    model.train()
    training_loss = 0.0
    for idx, (data, targets) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        targets = targets.float()  # Ensure targets are float for contrastive loss
        
        miner_out = miner(embeddings, targets)
        loss = loss_func(embeddings, targets, miner_out)
        
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        if idx % 20 == 0:
            num = miner.num_pos_pairs+miner.num_neg_pairs if type_model == "siamese" else miner.num_triplets
            print("Iteration {}: Loss = {}, Number of mined triplets = {}".format(idx, loss, num))
            break
    
    training_loss /= len(dataloader.dataset)
    return training_loss

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(model, train_set, test_set, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels, train_embeddings, train_labels, False)
    return accuracies