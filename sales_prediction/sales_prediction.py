import torch
from torch import nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
import numpy as np
from sklearn.metrics import r2_score

class Network (nn.Module):
    def __init__(self, aggregated, shop):
        super().__init__()
        
        n_neuroni = (11 if aggregated == 0
                     else 10)
        if shop == 0:
            n_neuroni -= 1
        
        self.vit = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        
        for param in self.vit.parameters():
            param.requires_grad = False
        self.image = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        self.tabular = nn.Sequential (
            nn.Linear(n_neuroni, n_neuroni),
            nn.ReLU(),
            nn.BatchNorm1d(n_neuroni)
        )
        
        self.description = nn.Sequential(
            nn.Linear(768, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300)
        )
        
        n_neuroni += 512 + 300
        
        self.final = nn.Sequential(
            nn.Linear(n_neuroni, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
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
        
    def forward(self, image, tab, desc):
        vit = self.vit(image)
        emb_image = self.image(vit)
        
        emb_tab = self.tabular(tab)
        
        emb_desc = self.description(desc)
       
        result = self.final(torch.cat((emb_image, emb_tab, emb_desc), 1))
        
        return result
    
def create_model(aggregated, shop):
    return Network(aggregated, shop)

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset) #quanti elementi nel dataset
    model.train() #setta il modello in modalità train: i pesi ora si modificano
    final_mse = 0
    
    for batch, (img, tab, desc, y) in enumerate(dataloader): #numero batch e coppia attributi-target. Quando si crea il dataloader dal dataset si dice già il mini batch
        img, desc, tab, y = img.to(device), desc.to(device), tab.to(device), y.to(device)

        pred = model(img, tab, desc)
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
    model.eval() #modalità valutazione, i pesi sono frizzati
    num_batches = len(dataloader)
    mae = nn.L1Loss()
    avg_mse, avg_mae, avg_r2 = 0, 0, 0
    label, prediction = np.array([]), np.array([])

    with torch.no_grad(): #si assicura che il gradiente qui non venga calcolato
        for img, tab, desc, y in dataloader:
            img, desc, tab, y = img.to(device), desc.to(device), tab.to(device), y.to(device)
            pred = model(img, tab, desc)
            
            y_np, pred_np = y.cpu().detach().numpy(), pred.cpu().detach().numpy()
            avg_mse += loss_fn(pred.squeeze(), y.float()).item()
            avg_mae += mae(pred.squeeze(), y.float()).item()
            avg_r2 += r2_score(y_np, pred_np)
            
            label = np.append(y_np, label)
            prediction = np.append(pred_np, prediction)

    avg_mse /= num_batches
    avg_mae /= num_batches
    avg_r2 /= num_batches
    bias = np.mean(prediction - label)
    
    print(f"Validation Error: \n Avg MSE: {avg_mse:>8f} \n Avg MAE: {avg_mae:>8f} \n Avg R2: {avg_r2:>8f}\n" + 
          f" Bias: {bias:>8f}\n")
    return avg_mse, avg_mae, avg_r2, bias

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.count = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.count = 0
        return self.early_stop
        