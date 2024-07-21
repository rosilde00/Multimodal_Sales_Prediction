import pandas as pd
import glob
import numpy as np
import shutil
import duckdb

#seleziona le immagini di cui si dispone delle vendite e setta gli id di aggregated nel giusto formato
aggr_path = 'C:\\ORS\\Data\\aggregated_sales.xlsx' 
aggr_dest_path = 'C:\\ORS\\Data\\aggregated_sales_final.xlsx'
img_path = 'C:\\ORS\\Data\\AllImages\\'
img_dest = 'C:\\ORS\\Data\\Images\\'
sales_path = 'C:\\ORS\\Data\\sales_anagrafica.xlsx'
sales_dest_path = 'C:\\ORS\\Data\\sales_anagrafica_final.csv'

data = pd.read_excel(aggr_path)
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

#id aggregated          
data['prodcode'] = new_ref
mask = ~data['prodcode'].str.endswith(('*.jpg')) #elimina i record dei prodotti di cui non c'è l'immagine (sono rimasti con la ref vecchia)
data = data[mask]
data.to_excel(aggr_dest_path, index=False)

#id join sales anagrafica
aggregated = data['prodcode']
sales = pd.read_excel(sales_path)

res = duckdb.query("SELECT * FROM data JOIN aggregated ON aggregated.prodcode LIKE REPLACE(data.idProdotto, '*', '%')").df()
res = res.drop(columns=['IdProdotto'], axis='columns')
res.to_csv(sales_dest_path, index=False)
