import pandas as pd
import numpy as np
import glob
from transformers import AutoTokenizer

def get_tabular(tabular_path):
    data, references, descriptions = get_data(tabular_path)
    tokenized_desc = word_embedding(descriptions)
    return data, references, tokenized_desc

def get_data(path):
    data = pd.read_csv(path)
    description = data['Descrizione'].to_list()
    description = list(map(lambda d: d.lower(), description))
    references = data['IdProdotto'].values
    data = data.drop(columns = ['Descrizione', 'IdProdotto'], axis='columns')
  
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
    
    return data, references, description

def word_embedding(description):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    tokenized_desc = tokenizer(description, padding = True, truncation = True, add_special_tokens = True, return_tensors="pt")
    return tokenized_desc.data