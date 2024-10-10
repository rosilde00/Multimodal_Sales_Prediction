import pandas as pd
import torch

def get_data(tabular_path, desc_path, img_path):
    data, references, target = get_tabular(tabular_path)
    descrizioni = torch.load(desc_path)
    immagini = torch.load(img_path)
    return data, references, immagini, descrizioni, target

def get_tabular(path):
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