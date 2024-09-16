import pandas as pd
from transformers import AutoModel
from transformers import AutoTokenizer
import torch

data = pd.read_csv('C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\no_negozi.csv')

descrizioni = list(data['Descrizione'].values)
descrizioni = list(map(lambda d: d.lower(), descrizioni))
id = list(data['IdProdotto'].values)

print('inizio trasformazione')

bert = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
tokenized_desc = tokenizer(descrizioni, padding = True, truncation = True, add_special_tokens = True, return_tensors="pt").data
bert_res = bert(**tokenized_desc)
desc_tensor = bert_res.last_hidden_state[:,0,:]

print('fine trasformazione')

desc_dict = {}
for i in range(0,len(id)):
    desc_dict[id[i]] = desc_tensor[i]

print(desc_dict['231_969_78976_03_10E_1.jpg'])

torch.save(desc_dict, 'desc.pt')