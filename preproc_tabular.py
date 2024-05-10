import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_tabular(path):
    data = pd.read_excel(path)
    references = data['CodiceArticoloColore']
    references = modify_ref(references)
    data = data.drop(['CodiceArticolo', 'CodiceArticoloColore', 'DescrizioneColore', 'WaveDescription'], axis='columns')
    
    encoded_labels, _ = pd.factorize(data['Colore'])
    data['Colore'] = encoded_labels
    encoded_labels, _ = pd.factorize(data['PianoTaglia'])
    data['PianoTaglia'] = encoded_labels
    encoded_labels, _ = pd.factorize(data['WaveCode'])
    data['WaveCode'] = encoded_labels 
    encoded_labels, _ = pd.factorize(data['AstronomicalSeasonDescription'])
    data['AstronomicalSeasonDescription'] = encoded_labels
    
    description_embedding = word_embedding(data['DescrizioneArticolo'])
    data['DescrizioneArticolo'] = description_embedding
    
    return data, references
    
def modify_ref(ref):
    for i in range(len(ref)):
        ref[i] = ref[i].replace('-', '_')
        ref[i] = ref[i][:7] + '_' + ref[i][7:]
        ref[i] = ref[i][:13] + '_' + ref[i][13:]
        ref[i] = ref[i] + '_'
        
    return ref

def word_embedding(descriptions):
    
    return 1
    