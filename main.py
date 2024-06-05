from preproc_tabular import get_tabular
from custom_dataset import getDataset
import neural_net
import torch
from torch import nn


target = [1,2,1,4,2,4,3,5,1,0] #TARGET FAKE
img_path = 'D:\\ORS\\Data\\ResizedImages\\'
tab_path = 'D:\\ORS\\Data\\prova.xlsx'
target_path = 'ciao'

data, references, descriptions, len_desc = get_tabular(img_path, tab_path)
train, val, test = getDataset(references, data, descriptions, target_path)
 
modello = neural_net.create_model(len_desc, len(data.columns))

learning_rate = 1e-3
batch_size = 1
epochs = 2
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(modello.parameters(), lr=learning_rate) 
early_stop = 3

stable_loss = 0
prec_loss = 20000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    neural_net.train_loop(train, modello, loss_fn, optimizer, 1)
    val_loss = neural_net.validation_loop(val, modello, loss_fn)
    stop, stable_loss = neural_net.early_stopping(val_loss, prec_loss, stable_loss, early_stop)
    if stop:
        break
print("Done!")