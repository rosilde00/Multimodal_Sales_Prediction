import torch
from torch import nn
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision.transforms import v2
import numpy as np

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class Network (nn.Module):
    def __init__(self, aggregated, shop):
        super().__init__()
        
        n_neuroni = (11 if aggregated == 0
                     else 10)
        if shop == 0:
            n_neuroni-=1

        self.vit = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.tabular = nn.Sequential(
            nn.Linear(n_neuroni, n_neuroni),
            nn.ReLU(),
            nn.BatchNorm1d(n_neuroni)
        )
        
        self.image = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        n_neuroni += 512
        
        self.final= nn.Sequential(
            nn.Linear(n_neuroni, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.3),
            nn.Linear(50, 1),
            nn.ReLU()
        )
        
        
    def forward(self, image, tab):
        vit = self.vit(image)
        emb_img = self.image(vit)
        
        emb_tab = self.tabular(tab)
        
        result = self.final(torch.cat((emb_img, emb_tab), 1))

        return result

class SalesDataset(Dataset):
    def __init__(self, references, tabular_data, img_path, target, transform=None, target_transform=None):
        self.img_ref = references 
        self.tabular = tabular_data
        self.target = target
        self.img_path = img_path
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        image = read_image(self.img_path + self.img_ref[idx], ImageReadMode.RGB)
        tabular_row = torch.from_numpy(self.tabular.iloc[idx].values).float()
        
        target = self.target[idx]
        
        if self.transform: 
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, tabular_row, target 
      
def create_model(aggregated, shop):
    return Network(aggregated, shop)

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset) 
    model.train() 
    final_mse = 0
    
    for batch, (img, tab, y) in enumerate(dataloader):
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        
        pred = model(img, tab)
        loss = loss_fn(pred.squeeze(), y.float())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() #resetta il gradiente (i numeri da aggiungere ai pesi?)

        final_mse += loss
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(tab)
            print(f"MSE: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    final_mse /= len(dataloader)
    print(f"In questa epoca, l'MSE medio è {final_mse}")

def validation_loop(dataloader, model, loss_fn, device):
    model.eval() 
    num_batches = len(dataloader)
    mae = nn.L1Loss()
    avg_mse, avg_mae = 0, 0
    label, prediction = np.array([]), np.array([])

    with torch.no_grad(): 
        for img, tab, y in dataloader:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
        
            pred = model(img, tab)
            
            y_np, pred_np = y.cpu().detach().numpy(), pred.cpu().detach().numpy()
            label = np.append(y_np, label)
            prediction = np.append(pred_np, prediction)
            
            avg_mse += loss_fn(pred.squeeze(), y.float()).item()
            avg_mae += mae(pred.squeeze(), y.float()).item()

    avg_mse /= num_batches
    avg_mae /= num_batches
    r2 = r2_score(label, prediction)
    bias = np.mean(prediction - label)
    
    print(f"Validation Error: \n Avg MSE: {avg_mse:>8f} \n Avg MAE: {avg_mae:>8f} \n R2: {r2:>8f}\n" + 
          f" Bias: {bias:>8f}\n")
    
    return avg_mse, avg_mae, r2, bias

def getDataset(references, tabular_data, target, img_path, batch_size, proportion):
    transform_img = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        
    dataset = SalesDataset(references, tabular_data, img_path, target, transform_img, None)
    
    partial, _ = random_split(dataset, [proportion, 1 - proportion])
    
    splitted_dataset = random_split(partial, [0.7, 0.3])
    train_dataloader = DataLoader(splitted_dataset[0], batch_size=batch_size)
    validation_dataloader = DataLoader(splitted_dataset[1], batch_size=batch_size)

    return train_dataloader, validation_dataloader