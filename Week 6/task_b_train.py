import os, sys
import torch
from torch import nn
from torchinfo import summary # to print model summary
from torchview import draw_graph # print model image
# from timeit import default_timer as timer  
import numpy as np

from train_utils import train
from data_utils import transform_data, loadImageData, myDataLoader
from plot_utils import save_loss_curves, plot_transformed_images, detail_one_sample_data
from task_b_model import VisionModel

# Note: this notebook requires torch >= 1.10.0
print("torch version: ", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("devide: ", device)


def walk_through_dir(data_path):
  for dirpath, dirnames, filenames in os.walk(data_path):
    filenames = [f for f in filenames if not f[0] == '.'] # to exclude the ".DS_Store"
    dirnames[:] = [d for d in dirnames if not d[0] == '.'] # to exclude the ".DS_Store"
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def get_data_sets_path(data_path):
    train_dir = os.path.join(data_path,"train")
    valid_dir = os.path.join(data_path,"valid")
    test_dir = os.path.join(data_path,"test")
    print("train dir: ", train_dir)
    print("valid dir: ", valid_dir)
    print("test dir: ", test_dir)
    return train_dir, valid_dir, test_dir 

# @torch.no_grad()
# def testSingleForwardPass(train_dataloader, model):
#     # 1. Get a batch of images and labels from the DataLoader
#     img_batch, label_batch = next(iter(train_dataloader))

#     # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
#     img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
#     print(f"Single image shape: {img_single.shape}\n")

#     # 3. Perform a forward pass on a single image
#     model.eval()
#     with torch.inference_mode():
#         pred = model(img_single.to(device))
        
#     # 4. Print out what's happening and convert model logits -> pred probs -> pred label
#     print(f"Output logits:\n{pred}\n")
#     print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
#     print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
#     print(f"Actual label:\n{label_single}")

@torch.no_grad()
def my_test_step_average(model, dataloader, test_data, output_folder, model_name):
    predictions = [['VideoName', 'ground_truth', 'prediction']]
    logits_dict = {}

    # Prepare the predictions list from filenames and categories
    for f in test_data.imgs:
        f_name = f[0]
        f_name = f_name[len(f_name)-19:len(f_name)-8] + '.mp4'
        cat = f[1] + 1  # f[1] from 0 to 6
        if f_name not in logits_dict:
            predictions.append([f_name, cat, '-1'])
            logits_dict[f_name] = []

    # Put model in test mode
    model.eval()
    
    print("Evaluating on test set...")
    for batch, (X, y) in enumerate(dataloader):
        # print(f"Batch: {batch} of {len(dataloader)}")
        X, y = X.to(device), y.to(device)

        # Forward pass to get logits and apply softmax
        y_logits = torch.softmax(model(X), dim=1)

        # Store softmax results in the dictionary under the corresponding filename
        video_name = test_data.imgs[batch][0]
        video_name = video_name[len(video_name)-19:len(video_name)-8] + '.mp4'
        logits_dict[video_name].append(y_logits.cpu().numpy())

    test_acc = 0
    # Aggregate logits per video name
    for idx, entry in enumerate(predictions[1:], start=1):
        video_name = entry[0]
        aggregated_logits = np.mean(logits_dict[video_name], axis=0)
        predictions[idx][2] = np.argmax(aggregated_logits) + 1
        test_acc += (predictions[idx][1] == predictions[idx][2])

    # Adjust metrics to get average accuracy per batch
    test_acc = test_acc / len(predictions[1:])
    print("Average accuracy = ", test_acc)

    # Save predictions to CSV
    np.savetxt(f"{output_folder}/predictions_test_set_{model_name}.csv", predictions, delimiter=",", fmt='%s')

@torch.no_grad()
def my_test_step(model, dataloader, test_data, output_folder, model_name):
    
    predictions = [['VideoName','ground_truth','prediction']]

    for f in test_data.imgs:
        f_name = f[0]
        f_name = f_name[len(f_name)-19:len(f_name)-4]+'.mp4'
        cat = f[1]+1 # f[1] from 0 to 6
        predictions.append([f_name,cat,'-1'])

    # Put model in test mode
    model.eval()
    
    # Setup test loss and test accuracy values
    test_acc = 0
    
    print("Evaluating on test set...")
    # Loop through data loader data batches
    counter = 1
    for batch, (X, y) in enumerate(dataloader):
        # print(f"Batch: {batch} of {len(dataloader)}")
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        test_acc += (y_pred_class == y).sum().item()/len(y_pred)

        predictions[counter][2] = y_pred_class.tolist()[0]+1 # y_pred_class.tolist()[0] from 0 to 6
        counter +=1

    # Adjust metrics to get average accuracy per batch 
    test_acc = test_acc / len(dataloader)
    print("Average accuracy = ", test_acc)

    np.savetxt(f"{output_folder}/predictions_test_set_{model_name}.csv", predictions, delimiter =",", fmt ='% s')


def main(data_path, model_stage, parameters_dict, class_weights, output_folder, checkpoints_folder, model_name):    

    # preliminaries
    walk_through_dir(data_path)
    train_dir, valid_dir, test_dir = get_data_sets_path(data_path)
    # image_path_list = print_image_samples(data_path, output_folder=output_folder, model_name=model_name)

    # data transformation
    data_train_transform, data_valid_test_transform = transform_data(parameters_dict['image_size'])
    # plot_transformed_images(image_path_list, transform=data_train_transform, output_folder=output_folder, model_name=model_name, n=3)
    
    # data loader
    train_data, valid_data, test_data, class_names = loadImageData(train_dir,valid_dir,test_dir,data_train_transform, data_valid_test_transform)
    num_classes = len(class_names)
    # detail_one_sample_data(train_data, class_names, output_folder, model_name=model_name)
    train_dataloader, valid_dataloader, test_dataloader = myDataLoader(train_data, valid_data, test_data, parameters_dict['num_workers'], parameters_dict['batch_size'], parameters_dict['batch_size_valid'])

    # model definition
    model = VisionModel(num_classes=num_classes).to(device)

    # model testing
    # testSingleForwardPass(train_dataloader, model)

    # saving the model image as "model.gv.png"
    model_graph = draw_graph(model, input_size=(1,3,224,224), expand_nested=True, directory=output_folder, filename=f"{model_name}_graph")
    model_graph.visual_graph.render(format='pdf')

    #
    # TRAIN
    #
    if model_stage=='train':

        # Unfreeze layers
        for name, para in model.named_parameters():
            para.requires_grad = True

        # do a test pass through of an example input size 
        summary(model, input_size=[1, 3, parameters_dict['image_size'][0], parameters_dict['image_size'][1]])

        # Setup loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=parameters_dict['learning_rate'])
        
        # Train model_0 
        model_results = train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=valid_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=parameters_dict['num_epochs'],
                            early_stop_thresh=parameters_dict['early_stopping'],
                            checkpoints_folder=checkpoints_folder,
                            model_name=model_name)

        save_loss_curves(model_results, output_folder=output_folder, model_name=model_name)

        # Test the model
        if parameters_dict['inference_strategy'] == 'average':
            my_test_step_average(model, test_dataloader, test_data, output_folder=output_folder, model_name=model_name)
        else:
            my_test_step(model, test_dataloader, test_data, output_folder=output_folder, model_name=model_name)

    elif model_stage == 'test':

        # load the model
        checkpoint = torch.load(f'{checkpoints_folder}/{model_name}_checkpoint.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        # evaluate on the test set and generate the 'predictions_test_set.csv' file (used later by the evaluation script)
        if parameters_dict['inference_strategy'] == 'average':
            my_test_step_average(model, test_dataloader, test_data, output_folder=output_folder, model_name=model_name)
        else:
            my_test_step(model, test_dataloader, test_data, output_folder=output_folder, model_name=model_name)
    

#------------------------------
# usage:
#
# python baseline_InceptionResnetV1.py ./data 'train' # train the model the first time
# python baseline_InceptionResnetV1.py ./data 'test' # evaluate on the test set
#
# where, './data' is the path to the input data with the 'train', 'valid' 'test' directories
#
if __name__ == '__main__': 

    parameters_dict = {
        'image_size': (224, 224),
        'num_workers': 4,
        'batch_size': 512,
        'batch_size_valid': 512,
        'num_epochs': 200,
        'learning_rate': 3e-6,
        'early_stopping': 50,
        'inference_strategy': 'average'
    }

    # Outputs folder
    output_folder = "c5/week6/outputs_task_b"

    # Checkpoints folder
    checkpoints_folder = "c5/week6/checkpoints"

    # Model name (used when saving plots, checkpoint, etc.)
    model_name = "visualModel_augmentation_average_new_weights"

    # train data distribution per category [10, 164, 1264, 2932, 1353, 232, 51]
    # class_weights = torch.tensor([3.0, 1.0, 0.25, 0.2, 0.25, 1.0, 1.5]).to(device) # can be changed to use class weights
    # class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device) # no class weights

    data_size_per_cat = torch.tensor([10, 164, 1264, 2932, 1353, 232, 51])
    total_samples = sum(data_size_per_cat)
    num_classes = len(data_size_per_cat)
    class_weights = total_samples / (num_classes * data_size_per_cat)
    class_weights = class_weights.to(device)
    print("class_weights: ", class_weights)
    
    data_path = sys.argv[1] # path to the input data
    model_stage = sys.argv[2] # 'train', 'resume' or 'test'

    main(data_path, model_stage, parameters_dict, class_weights, output_folder, checkpoints_folder, model_name)

