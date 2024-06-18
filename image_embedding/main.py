from image_embedding.dataset_embedding import getDataset
import image_embedding.image_embedding
import torch
from torch import nn
import pandas as pd


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) #presi da timm
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

img_path = 'D:\\ORS\\Data\\ResizedImages\\'
tab_path = 'D:\\ORS\\Data\\aggregated_sales_final.xlxs'

data = pd.read_excel(tab_path)
train, val, test = getDataset(data, img_path)

modello = image_embedding.create_model()

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
    image_embedding.train_loop(train, modello, loss_fn, optimizer, 1)
    val_loss = image_embedding.validation_loop(val, modello, loss_fn)
    stop, stable_loss = image_embedding.early_stopping(val_loss, prec_loss, stable_loss, early_stop)
    if stop:
        break
print("Done!")