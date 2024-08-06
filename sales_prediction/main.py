from sales_prediction.preproc_tabular import get_tabular
from sales_prediction.dataset_sales import getDataset
import sales_prediction.sales_prediction as sp
from torch.optim.adamw import AdamW
from torch import nn
import torch
import os

img_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\ResizedImages\\'
tab_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\sales_anagrafica_final.csv'

data, references, descriptions = get_tabular(tab_path)
train, val = getDataset(references, data, descriptions, img_path)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Sto usando {device}")

modello = sp.create_model().to(device)

batch_size = 64
epochs = 15
criterion = nn.MSELoss()
optimizer = AdamW(modello.parameters(), lr=1e-3) 
early_stop = sp.EarlyStopping(5, 0)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    sp.train_loop(train, modello, criterion, optimizer, batch_size, device)
    val_loss = sp.validation_loop(val, modello, criterion)
    early_stp = early_stop(val_loss)
    if early_stp:
        print('Early Stop attivato.')
        break
    
print("Done!")

cartella = input("Inserire il nome della cartella in cui salvare i pesi.")
os.makedirs(cartella)
descrizione = input("Inserire la descrizione dei pesi salvati.")
with open(f'.\\results\\{cartella}\\descrizione.txt', 'w') as file:
    file.write(descrizione)
torch.save(modello, f'.\\results\\{cartella}\\weights.pth')