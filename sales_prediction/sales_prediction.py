import torch
from torch import nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
from transformers import AutoModel

class Network (nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Linear(50, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
        )
        
        self.tabular = nn.Sequential (
            nn.Linear(12, 12),
            nn.BatchNorm1d(12),
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
            nn.Linear(50, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
        )
        
        self.final = nn.Sequential(
            nn.Linear(36, 10),
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
    
def create_model():
    return Network()

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
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(dataloader, model, loss_fn):
    model.eval() #modalità valutazione, i pesi sono frizzati
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad(): #si assicura che il gradiente qui non venga calcolato
        for img, tab, desc, y in dataloader:
            pred = model(img, tab, desc)
            val_loss += loss_fn(pred.squeeze(), y.float()).item()

    val_loss /= num_batches
    print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")
    return val_loss

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
        