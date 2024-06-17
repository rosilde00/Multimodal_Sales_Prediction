import glob
import pandas as pd
import numpy as np

img_path = 'C:\\ORS\\Data\\ResizedImages\\'
data_path = 'C:\\ORS\\Data\\sales_anagrafica.xlsx'
dest_path = 'C:\\ORS\\Data\\sales_anagrafica_final.xlsx'

data = pd.read_excel(data_path)
references = data['IdProdotto'].values
new_ref = list(references)
for ref in list(references):
    images = glob.glob(img_path + ref)
    if len(images) != 0:
        idx = new_ref.index(ref)
        new_ref.remove(ref)
        new_ref = new_ref[:idx] + images + new_ref[idx:]

        times = len(images)-1
        if times != 0:
            new_data = data.values
            new_data = np.insert(data.values, idx, [new_data[idx]]*times, axis=0)
            new_data = pd.DataFrame(new_data)
            new_data.columns = data.columns
            data = new_data
            
data['IdProdotto'] = new_ref
data.to_excel(dest_path, index=False)