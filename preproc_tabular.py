import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer

def get_tabular(path):
    data = pd.read_excel(path)
    references = modify_ref(data['Stagione'], data['CodiceArticolo'], data['CodiceColore'])
    description= data['Descrizione']
    description_embedding = word_embedding(description)
    data = data.drop(['Stagione', 'CodiceArticolo', 'Descrizione', 'DescrizioneColore', 'AreaDescription', 
                      'CategoryDescription', 'SectorDescription', 'DepartmentDescription', 'WaveDescription',
                      'AstronomicalSeasonDescription', 'SalesSeasonBeginDate', 'SalesSeasonEndDate'], axis='columns')
    
    for col in data.columns:
        encoded_labels, _ = pd.factorize(data[col])
        if encoded_labels.std() != 0:
            encoded_labels = (encoded_labels - encoded_labels.mean())/encoded_labels.std()
        data[col] = encoded_labels
    
    return data, references, description_embedding #data è un dataframe, references lista di stringhe, desc_emb lista di tensori
    
def modify_ref(season, catr, ccol):
    new_ref = list()
    for sn, ca, cc in zip(season, catr, ccol):
        sn = str(sn)
        ca = str(ca)
        cc = str(cc)
        r = sn + '_' + ca[:3] + '_' + ca[3:8] + '_' + ca[8:] + '_' + cc + '_*.jpg'
        new_ref.append(r)
    return new_ref

def get_dictionary(descriptions):
    total_text = ""
    for d in descriptions:
        total_text = total_text + " " + d
    
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(total_text)
    tokens_set = set(tokens)
    dictionary = {word: j for j, word in enumerate(tokens_set)}
    return dictionary

def word_embedding(descriptions):
    dictionary = get_dictionary(descriptions)
    tokenizer = get_tokenizer('basic_english')
    embed_layer = nn.Embedding(len(dictionary), 10)  

    tensor_list = []
    for d in descriptions:
        tokens = tokenizer(d)
        desc_tensor = torch.zeros(1,10)
        for t in tokens:
            lookup_tensor = torch.tensor([dictionary[t]], dtype=torch.long)
            embed = embed_layer(lookup_tensor)
            desc_tensor = torch.cat((desc_tensor, embed), 0)
        desc_tensor = desc_tensor[1:]
        tensor_list.append(desc_tensor)
        
    return tensor_list
    
def duplicate_row(img_dir, data, descriptions, references):
    new_ref = list(references)
    for ref in references:
        images = glob.glob(img_dir + ref)
        if len(images) != 0:
            idx = new_ref.index(ref)
            new_ref.remove(ref)
            new_ref = new_ref[:idx] + images + new_ref[idx:]

            times = len(images)-1
            if times != 0:
                descriptions = descriptions[:idx] + [descriptions[idx]]*times + descriptions[idx:]
                new_data = data.values
                new_data = np.insert(data.values, idx, [new_data[idx]]*times, axis=0)
                new_data = pd.DataFrame(new_data)
                new_data.columns = data.columns
                data = new_data
            
    return new_data, descriptions, new_ref