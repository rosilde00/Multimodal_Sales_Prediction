from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import os 
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

target = [1,2,1,4,2,4,3,5,1,0] #TARGET FAKE

class CustomDataset(Dataset):
    def __init__(self, img_dir, tabular_data, descriptions, target_file, image_transform=None, target_transform=None):
        self.img_dir = img_dir 
        self.tabular = tabular_data
        self.descriptions = descriptions
        self.target = target ###
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx): #ritorna coppia immagine riga e label
        tabular_row = self.tabular.iloc[idx]
        img_path = os.path.join(self.img_dir, self.tabular.iloc[idx, 0])
        image = read_image(img_path)
        #label = self.img_labels.iloc[idx, 1] QUANDO CI SARA IL TARGET
        label = target[idx]
        if self.transform: 
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, tabular_row, label

def getDataset(target_file, img_dir, tabular_data,):
    #inserire transform
    dataset = CustomDataset(target_file, img_dir, tabular_data, None, None)
    splitted_dataset = random_split(dataset, [0.6, 0.3, 0.1])

    train_dataloader = DataLoader(splitted_dataset[0], batch_size=64)
    validation_dataloader = DataLoader(splitted_dataset[1], batch_size=64)
    test_dataloader = DataLoader(splitted_dataset[2], batch_size=64)
    return train_dataloader, validation_dataloader, test_dataloader


#transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#])