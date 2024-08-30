import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class Network (nn.Module):
    def __init__(self, aggregated):
        super().__init__()
        
        n_neuroni = (11 if aggregated == 0
                     else 10)
        
        self.tabular = nn.Sequential (
            nn.Linear(n_neuroni, n_neuroni),
            nn.BatchNorm1d(n_neuroni),
            nn.ReLU(),
            nn.Linear(n_neuroni, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        
    def forward(self, tab):
        result = self.tabular(tab)
        return result

class SalesDataset(Dataset):
    def __init__(self, tabular_data, target):
        self.tabular = tabular_data
        self.target = target

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        tabular_row = torch.from_numpy(self.tabular.iloc[idx].values).float()
        target = self.target[idx]
        
        return tabular_row, target 
      
def create_model(aggregated):
    return Network(aggregated)

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset) 
    model.train() 
    for batch, (tab, y) in enumerate(dataloader):
        tab, y = tab.to(device), y.to(device)
        
        pred = model(tab)
        loss = loss_fn(pred.squeeze(), y.float())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(tab)
            print(f"MSE: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(dataloader, model, loss_fn, device):
    model.eval() #modalità valutazione, i pesi sono frizzati
    num_batches = len(dataloader)
    avg_mse = 0
    avg_mae = 0
    avg_r2 = 0

    with torch.no_grad(): #si assicura che il gradiente qui non venga calcolato
        for tab, y in dataloader:
            tab, y = tab.to(device), y.to(device)
        
            pred = model(tab)
            avg_mse += loss_fn(pred.squeeze(), y.float()).item()
            avg_mae += mean_absolute_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
            avg_r2 += r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())

    avg_mse /= num_batches
    avg_mae /= num_batches
    avg_r2 /= num_batches
    
    print(f"Validation Error: \n Avg MSE: {avg_mse:>8f} \n Avg MAE: {avg_mae:>8f} \n Avg R2: {avg_r2}\n")
    return avg_mse, avg_mae, avg_r2


def getDataset(tabular_data, target, batch_size, proportion):
    dataset = SalesDataset(tabular_data, target)
    
    partial, _ = random_split(dataset, [proportion, 1 - proportion])
    
    splitted_dataset = random_split(partial, [0.7, 0.3])
    train_dataloader = DataLoader(splitted_dataset[0], batch_size=batch_size)
    validation_dataloader = DataLoader(splitted_dataset[1], batch_size=batch_size)

    return train_dataloader, validation_dataloader