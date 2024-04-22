from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

def transform_data(IMAGE_SIZE):

    data_train_transform = transforms.Compose([
        v2.Resize(size=IMAGE_SIZE),
        v2.TrivialAugmentWide(),
        # Turn the image into a torch.Tensor
        v2.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        # resnet50 normalization
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_valid_test_transform = transforms.Compose([
        # Resize the images to IMAGE_SIZE xIMAGE_SIZE 
        v2.Resize(size=IMAGE_SIZE),
        # Turn the image into a torch.Tensor
        v2.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        # resnet50 normalization
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    return data_train_transform, data_valid_test_transform

def loadImageData(train_dir, valid_dir, test_dir, data_train_transform, data_valid_test_transform):
    # Creating training set
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=data_train_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    # Creating validation set
    valid_data = datasets.ImageFolder(root=valid_dir, # target folder of images
                                    transform=data_valid_test_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    #Creating test set
    test_data = datasets.ImageFolder(root=test_dir, transform=data_valid_test_transform)

    print(f"Train data:\n{train_data}\nValidation data:\n{valid_data}\nTest data:\n{test_data}")

    # Get class names as a list
    class_names = train_data.classes
    print("Class names: ",class_names)

    # Check the lengths
    print("The lengths of the training, validation and test sets: ", len(train_data), len(valid_data), len(test_data))  

    return train_data, valid_data, test_data, class_names

def myDataLoader(train_data, valid_data, test_data, NUM_WORKERS, BATCH_SIZE, BATCH_SIZE_VALID):

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    # Turn train and test Datasets into DataLoaders
    valid_dataloader = DataLoader(dataset=valid_data, 
                                batch_size=BATCH_SIZE_VALID, # how many samples per batch?
                                num_workers=NUM_WORKERS,
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=1, 
                                num_workers=NUM_WORKERS,
                                shuffle=False) # don't usually need to shuffle testing data

    # Now let's get a batch image and check the shape of this batch.    
    img, label = next(iter(train_dataloader))

    # Note that batch size will now be 1.  
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    return train_dataloader, valid_dataloader, test_dataloader