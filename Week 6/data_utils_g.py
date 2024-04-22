import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchaudio
import torchaudio_augmentations as ta
import os, pickle, random, re, time

def transform_image_data(IMAGE_SIZE):

    data_train_transform = transforms.Compose([
        v2.Resize(size=IMAGE_SIZE),
        v2.TrivialAugmentWide(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_valid_test_transform = transforms.Compose([
        v2.Resize(size=IMAGE_SIZE),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    return data_train_transform, data_valid_test_transform

def transform_audio_data():

    num_samples = 16000 * 10
    data_train_transform = ta.Compose(transforms=[
        torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000),
        ta.RandomResizedCrop(n_samples=num_samples),
        ta.RandomApply([ta.Noise(min_snr=0.001, max_snr=0.005)], p=0.25),
        ta.RandomApply([ta.Gain()], p=0.25),
        ta.RandomApply([ta.HighLowPass(sample_rate=16000)], p=0.25),
        ta.RandomApply([ta.PitchShift(n_samples=num_samples, sample_rate=16000)], p=0.25),
        ta.RandomApply([ta.Reverb(sample_rate=16000)], p=0.25),
    ])

    data_valid_test_transform = ta.Compose(transforms=[
        torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000),
    ])

    return data_train_transform, data_valid_test_transform

class MultiModalDataset(Dataset):
    def __init__(self, directory, image_transform=None, audio_transform=None, replace_token=''):
        # Load all image data and get labels and file paths
        self.image_dataset = datasets.ImageFolder(root=directory, transform=image_transform)
        self.audio_transform = audio_transform
        self.directory = directory
        self.replace_token = replace_token

    def __len__(self):
        return len(self.image_dataset)
    
    def preprocess_text(self, data):
        pattern = r"\[inaudible \d{2}:\d{2}:\d{2}\]"
        for i in range(len(data)):
            data[i] = re.sub(pattern, self.replace_token, data[i])
        return data

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        img_name = self.image_dataset.imgs[idx][0]  # Full path of the image

        base_path = os.path.splitext(img_name)[0]  # Remove the image extension

        # Load audio
        audio_path = base_path + '.wav'
        audio, _ = torchaudio.load(audio_path)
        if self.audio_transform:
            if audio.shape[1] < 122416:
                audio = torch.cat([audio, torch.zeros(1, 122416 - audio.shape[1])], dim=1)
            audio = self.audio_transform(audio).squeeze(0)

        # Load text
        text_path = base_path + '.pkl'
        with open(text_path, 'rb') as file:
            text = pickle.load(file)
        text = self.preprocess_text([text])[0]

        return image, audio, text, label

def loadImageData(train_dir, valid_dir, test_dir, image_size=(224, 224), replace_token=''):

    data_train_image_transform, data_valid_test_image_transform = transform_image_data(image_size)
    data_train_audio_transform, data_valid_test_audio_transform = transform_audio_data()
    
    train_data = MultiModalDataset(directory=train_dir, image_transform=data_train_image_transform, audio_transform=data_train_audio_transform, replace_token=replace_token)
    valid_data = MultiModalDataset(directory=valid_dir, image_transform=data_valid_test_image_transform, audio_transform=data_valid_test_audio_transform, replace_token=replace_token)
    test_data = MultiModalDataset(directory=test_dir, image_transform=data_valid_test_image_transform, audio_transform=data_valid_test_audio_transform, replace_token=replace_token)

    return train_data, valid_data, test_data, ['1', '2', '3', '4', '5', '6', '7']

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

    return train_dataloader, valid_dataloader, test_dataloader