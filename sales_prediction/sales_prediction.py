import torch
from torch import nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
from transformers import AutoModel
from sklearn.metrics import mean_absolute_error, r2_score

class Network (nn.Module):
    def __init__(self, aggregated):
        super().__init__()
        
        n_neuroni = (12 if aggregated == 0
                     else 11)
        
        self.vit = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.img_emb= nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, n_neuroni),
            nn.BatchNorm1d(n_neuroni),
            nn.ReLU(),
        )
        
        self.tabular = nn.Sequential (
            nn.Linear(n_neuroni, n_neuroni),
            nn.BatchNorm1d(n_neuroni),
            nn.ReLU(),
        )
        
        self.descriptions = nn.Sequential(
            nn.Linear(768, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, n_neuroni),
            nn.BatchNorm1d(n_neuroni),
            nn.ReLU(),
        )
        
        n_neuroni *= 3
        
        self.final = nn.Sequential(
            nn.Linear(n_neuroni, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        
    def forward(self, image, tab, desc_tensor):
        vit = self.vit(image)
        emb_image = self.img_emb(vit)
        
        emb_tab = self.tabular(tab)
        
        emb_desc = self.descriptions(desc_tensor)
        
        result = self.final(torch.cat((emb_image, emb_tab, emb_desc), 1))
        
        return result
    
def create_model(aggregated):
    return Network(aggregated)

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset) #quanti elementi nel dataset
    model.train() #setta il modello in modalità train: i pesi ora si modificano
    for batch, (img, tab, desc, y) in enumerate(dataloader): #numero batch e coppia attributi-target. Quando si crea il dataloader dal dataset si dice già il mini batch
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        
        bert = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        bert_res = bert(**desc)
        desc_tensor = bert_res.last_hidden_state[:,0,:]
        desc_tensor = desc_tensor.to(device)
        
        pred = model(img, tab, desc_tensor)
        loss = loss_fn(pred.squeeze(), y.float())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() #resetta il gradiente (i numeri da aggiungere ai pesi?)

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
        for img, tab, desc, y in dataloader:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
        
            bert = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
            bert_res = bert(**desc)
            desc_tensor = bert_res.last_hidden_state[:,0,:]
            desc_tensor = desc_tensor.to(device)
        
            pred = model(img, tab, desc_tensor)
            avg_mse += loss_fn(pred.squeeze(), y.float()).item()
            avg_mae += mean_absolute_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
            avg_r2 += r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())

    avg_mse /= num_batches
    avg_mae /= num_batches
    avg_r2 /= num_batches
    
    print(f"Validation Error: \n Avg MSE: {avg_mse:>8f} \n Avg MAE: {avg_mae:>8f} \n Avg R2: {avg_r2}\n")
    return avg_mse, avg_mae, avg_r2

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
        