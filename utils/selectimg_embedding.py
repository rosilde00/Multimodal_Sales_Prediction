import pandas as pd
import glob
import numpy as np
import shutil
#seleziona le immagini nell'aggregated e mette gli id giusti
path = 'C:\\ORS\\Data\\aggregated_sales.xlsx' 
dest_path = 'C:\\ORS\\Data\\aggregated_sales_final.xlsx'
img_path = 'C:\\ORS\\Data\\222_1_di_2\\222_1_di_2\\'
img_dest = 'C:\\ORS\\Data\\Images\\'

data = pd.read_excel(path)
references = data['prodcode'].values
new_ref = list(references)
for ref in list(references):
    images = glob.glob(img_path + ref)
    
    if len(images) != 0:
        images = list(map(lambda img: img[len(img_path):], images))
    
        idx = new_ref.index(ref)
        new_ref.remove(ref)
        new_ref = new_ref[:idx] + images + new_ref[idx:]
        for img in images:
            shutil.copy(img_path + img, img_dest + img)

        times = len(images)-1
        if times != 0:
            new_data = data.values
            new_data = np.insert(data.values, idx, [new_data[idx]]*times, axis=0)
            new_data = pd.DataFrame(new_data)
            new_data.columns = data.columns
            data = new_data
            
data['prodcode'] = new_ref
data.to_excel(dest_path, index=False)