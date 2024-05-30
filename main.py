import preproc_tabular
from custom_dataset import getDataset

target = [1,2,1,4,2,4,3,5,1,0] #TARGET FAKE
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406) #presi da timm
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
img_path = 'D:\\ORS\\Data\\ResizedImages\\'
tab_path = 'D:\\ORS\\Data\\Anagrafica.xlsx'

data, references, descriptions = preproc_tabular.get_tabular('D:\\ORS\\Data\\prova.xlsx')
newdata, newdescription, newreferences = preproc_tabular.duplicate_row('D:\\ORS\\Data\\Images\\', data, descriptions, references)
train, val, test = getDataset(newreferences, newdata, newdescription, "ciao")