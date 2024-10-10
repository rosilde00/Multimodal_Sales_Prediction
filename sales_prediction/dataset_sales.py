from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision.io import ImageReadMode

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) #presi da timm
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class SalesDataset(Dataset):
    def __init__(self, references, tabular_data, images, descriptions, target):
        self.references = references 
        self.tabular = tabular_data
        self.images = images
        self.descriptions = descriptions
        self.target = target

    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, idx):
        img_tensor = self.images[self.references[idx]]
        
        tabular_row = torch.from_numpy(self.tabular.iloc[idx].values).float()

        desc_tensor = self.descriptions[self.references[idx]]
        
        target = self.target[idx]
        
        return img_tensor, tabular_row, desc_tensor, target 

def getDataset(references, tabular_data, images, descriptions, target, batch_size, proportion):
    dataset = SalesDataset(references, tabular_data, images, descriptions, target)
    
    partial, _ = random_split(dataset, [proportion, 1 - proportion])
    
    splitted_dataset = random_split(partial, [0.7, 0.3])
    train_dataloader = DataLoader(splitted_dataset[0], batch_size=batch_size)
    validation_dataloader = DataLoader(splitted_dataset[1], batch_size=batch_size)

    return train_dataloader, validation_dataloader
