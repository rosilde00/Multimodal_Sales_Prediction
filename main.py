import preproc_tabular
from custom_dataset import getDataset
import neural_net
import torch
from torch import nn

target = [1,2,1,4,2,4,3,5,1,0] #TARGET FAKE
img_path = 'D:\\ORS\\Data\\ResizedImages\\'
tab_path = 'D:\\ORS\\Data\\prova.xlsx'

data, references, descriptions = preproc_tabular.get_tabular('D:\\ORS\\Data\\prova.xlsx')
newdata, newdescription, newreferences = preproc_tabular.duplicate_row(img_path, data, descriptions, references)
train, val, test = getDataset(newreferences, newdata, newdescription, "ciao")

 
modello = neural_net.create_model()

learning_rate = 1e-3
batch_size = 1
epochs = 2
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(modello.parameters(), lr=learning_rate) 


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    neural_net.train_loop(train, modello, loss_fn, optimizer, 1)
    neural_net.validation_loop(val, modello, loss_fn)
print("Done!")