from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
import torch
from torch import nn

vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)

class Network (nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1) #pesi pre addestrati?
        self.linear = nn.Sequential (
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.ReLU(),
        )
        
    def forward(self, image):
        emb_image = self.img(image)
        pred = self.linear(emb_image)
        return pred
    
def create_model():
    return Network()

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset) 
    model.train() 
    for batch, (img, y) in enumerate(dataloader): 
        pred = model(img)
        loss = loss_fn(pred.squeeze(), y.float())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        loss = loss.item()
        print(f"loss: {loss:>7f}")


def validation_loop(dataloader, model, loss_fn):
    model.eval() 
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad(): 
        for img, y in dataloader:
            pred = model(img)
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
        