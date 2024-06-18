from sales_prediction.preproc_tabular import get_tabular
from sales_prediction.dataset_sales import getDataset
import sales_prediction.sales_prediction as sales_prediction
import torch
from torch import nn


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) #presi da timm
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

img_path = 'D:\\ORS\\Data\\ResizedImages\\'
tab_path = 'D:\\ORS\\Data\\sales_anagrafica_final.xlxs'
target_path = 'ciao'

data, descriptions, references = get_tabular(img_path, tab_path)
train, val, test = getDataset(references, data, descriptions, target_path)

modello = sales_prediction.create_model()

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
    sales_prediction.train_loop(train, modello, loss_fn, optimizer, 1)
    val_loss = sales_prediction.validation_loop(val, modello, loss_fn)
    stop, stable_loss = sales_prediction.early_stopping(val_loss, prec_loss, stable_loss, early_stop)
    if stop:
        break
print("Done!")