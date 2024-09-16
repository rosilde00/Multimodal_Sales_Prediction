import pandas as pd
from transformers import AutoTokenizer
import torch

def get_tabular(tabular_path, desc_path):
    data, references, target = get_data(tabular_path)
    #tokenized_desc = word_embedding(descriptions)
    descrizioni = torch.load(desc_path)
    return data, references, descrizioni, target

def get_data(path):
    data = pd.read_csv(path)
    references = data['IdProdotto'].values
    target = data['Quantity'].values
    
    data = data.drop(columns = ['Descrizione', 'IdProdotto', 'Quantity'], axis='columns')
  
    columns = ['CodiceColore', 'PianoTaglia', 'WaveCode', 'AstronomicalSeasonExternalID', 'SalesSeasonDescription']
    for col in columns:
        encoded_labels, _ = pd.factorize(data[col])
        data[col] = encoded_labels
    
    for col in data.columns: #normalizzo tutto tranne la quantità
        if col != 'Quantity':
            val = data[col]
            if val.std() != 0:
                normalized_labels = (val - val.mean())/val.std()
                data[col] = normalized_labels

    return data, references, target

def word_embedding(description):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    tokenized_desc = tokenizer(description, padding = True, truncation = True, add_special_tokens = True, return_tensors="pt")
    return tokenized_desc.data