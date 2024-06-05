import pandas as pd
import numpy as np
import glob
from transformers import AutoTokenizer

def get_data(path):
    data = pd.read_excel(path)
    references = modify_ref(data['Stagione'], data['CodiceArticolo'], data['CodiceColore'])
    description = data['Descrizione'].to_list()
    data = data.drop(['Stagione', 'CodiceArticolo', 'Descrizione', 'DescrizioneColore', 'AreaDescription', 
                      'CategoryDescription', 'SectorDescription', 'DepartmentDescription', 'WaveDescription',
                      'AstronomicalSeasonDescription', 'SalesSeasonBeginDate', 'SalesSeasonEndDate'], axis='columns')
    
    for col in data.columns:
        encoded_labels, _ = pd.factorize(data[col])
        if encoded_labels.std() != 0:
            encoded_labels = (encoded_labels - encoded_labels.mean())/encoded_labels.std()
        data[col] = encoded_labels
    
    return data, references, description #data è un dataframe, references lista di stringhe, desc_emb lista di tensori

def duplicate_row(img_dir, data, references, descriptions):
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

    
def modify_ref(season, catr, ccol):
    new_ref = list()
    for sn, ca, cc in zip(season, catr, ccol):
        sn, ca, cc = str(sn), str(ca), str(cc)
        r = sn + '_' + ca[:3] + '_' + ca[3:8] + '_' + ca[8:] + '_' + cc + '_*.jpg'
        new_ref.append(r)
    return new_ref

def word_embedding(descriptions):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_desc = tokenizer(descriptions, padding = True, truncation = False, return_tensors="pt")
    return tokenized_desc.data
    
def get_tabular(img_dir, tabular_path):
    data, references, descriptions = get_data(tabular_path)
    newdata, newdescription, newreferences = duplicate_row(img_dir, data, references, descriptions)
    tokenized_desc = word_embedding(newdescription)
    return newdata, tokenized_desc, newreferences