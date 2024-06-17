import torch
from torch import nn
from transformers import AutoModel

class Network (nn.Module):
    def __init__(self):
        super().__init__()
        self.img = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 12),
            nn.ReLU(),
        ) 
        self.tabular = nn.Sequential (
            nn.Linear(12, 12),
            nn.ReLU(),
        )
        self.bert = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        self.descriptions = nn.Sequential(
            nn.Linear(768, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 12),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(36, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU()
        )
        
    def forward(self, image, tab, desc):
        emb_image = self.img(image)
        emb_tab = self.tabular(tab)
        bert_res = self.bert(**desc)
        last_hidden = bert_res.last_hidden_state[:,0,:]
        emb_desc = self.descriptions(last_hidden)
        result = self.final(torch.cat((emb_image, emb_tab, emb_desc), 1))
        return result
    
def create_model():
    return Network()

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset) #quanti elementi nel dataset
    model.train() #setta il modello in modalità train: i pesi ora si modificano
    for batch, (img, tab, desc, y) in enumerate(dataloader): #numero batch e coppia attributi-target. Quando si crea il dataloader dal dataset si dice già il mini batch
        pred = model(img, tab, desc)
        loss = loss_fn(pred.squeeze(), y.float())
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() #resetta il gradiente (i numeri da aggiungere ai pesi?)

        loss = loss.item() #per outputtare il risultato al momento ******CAMBIARE GUARDARE PASTICCI******
        print(f"loss: {loss:>7f}")


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

def early_stopping (actual, previuos, n_epoch, limit):
    stop = False
    if actual < previuos:
        new_nepoch = 0
    else:
        new_nepoch = n_epoch + 1
        if new_nepoch == limit:
            stop = True
    return stop, new_nepoch
        