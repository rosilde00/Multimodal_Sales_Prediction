from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision.io import ImageReadMode

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) #presi da timm
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CustomDataset(Dataset):
    def __init__(self, references, tabular_data, descriptions, img_path, transform=None, target_transform=None):
        self.img_ref = references 
        self.tabular = tabular_data
        self.descriptions = descriptions
        self.target = tabular_data['Quantity'].values
        self.img_path = img_path
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_ref)
    
    def __getitem__(self, idx):
        image = read_image(self.img_path + self.img_ref[idx], ImageReadMode.RGB)
        tabular_row = torch.from_numpy(self.tabular.iloc[idx].values).float()
        
        description = dict() #le descrizioni sono un dizionario con input_ids [1, 3, 2, 1, 6] e maschera [1, 1, 1, 0, 0]
        token_tensor = self.descriptions.get('input_ids')
        description['input_ids'] = token_tensor[idx]
        mask_tensor = self.descriptions.get('attention_mask')
        description['attention_mask'] = mask_tensor[idx]
        
        target = self.target[idx]
        
        if self.transform: 
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, tabular_row, description, target 

def getDataset(references, tabular_data, descriptions, img_path):
    transform_img = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    dataset = CustomDataset(references, tabular_data, descriptions, img_path, transform_img, None)
    
    partial, _ = random_split(dataset, [0.10, 0.90])
    
    splitted_dataset = random_split(partial, [0.7, 0.3])
    train_dataloader = DataLoader(splitted_dataset[0], batch_size=64)
    validation_dataloader = DataLoader(splitted_dataset[1], batch_size=64)
    #test_dataloader = DataLoader(splitted_dataset[2], batch_size=64)
    return train_dataloader, validation_dataloader
