import matplotlib.pyplot as plt
import random
from PIL import Image
import glob
from pathlib import Path
import numpy as np

def detail_one_sample_data(train_data, class_names, output_folder, model_name):
    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    # Rearrange the order of dimensions
    img_permute = img.permute(1, 2, 0)

    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    plt.figure(figsize=(10, 7))
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(class_names[label], fontsize=14);    
    plt.savefig(f"{output_folder}/sample_data_detailed_{model_name}.jpg")

def plot_transformed_images(image_paths, transform, output_folder, model_name, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for c, image_path in enumerate(random_image_paths):
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            plt.savefig(f"{output_folder}/transformed_image_{model_name}.jpg")

def print_image_samples(data_path, output_folder, model_name):
    # Set seed
    random.seed(42) 

    # 1. Get all image paths (* means "any combination")
    image_path_list= glob.glob(f"{data_path}/*/*/*.jpg")

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = Path(random_image_path).parent.stem

    # 4. Open image
    img = Image.open(random_image_path)

    # 5. Print metadata
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")
    print(img)

    # Turn the image into an array
    img_as_array = np.asarray(img)

    # Plot the image with matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(img_as_array)
    plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    plt.axis(False)
    plt.savefig(f"{output_folder}/sample_data_intro_{model_name}.jpg")
    print("sample image saved as sample_data_intro.jpg")
    
    return image_path_list

def save_loss_curves(model_results, output_folder, model_name):
  
    results = dict(list(model_results.items()))

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f"{output_folder}/training_history_{model_name}.jpg")