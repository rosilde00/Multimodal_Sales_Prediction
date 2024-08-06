import pandas as pd
import glob
import shutil
import duckdb

#seleziona le immagini di cui si dispone delle vendite e setta gli id di aggregated nel giusto formato
img_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\AllImages\\'
img_dest = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\Images\\'
sales_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\sales_anagrafica.xlsx'
sales_dest_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\sales_anagrafica_final.csv'

data = pd.read_excel(sales_path)
references = set(data['IdProdotto'])

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

aggregated = pd.DataFrame(new_ref, columns=['IdProdotto'])
data = data.rename(columns={'IdProdotto': 'ref'})       
res = duckdb.query("SELECT * FROM data JOIN aggregated ON aggregated.IdProdotto LIKE REPLACE(data.ref, '*', '%')").df()
res = res.drop(columns=['ref'], axis='columns')
mask = ~res['IdProdotto'].str.endswith(('*.jpg')) #elimina i record dei prodotti di cui non c'è l'immagine (sono rimasti con la ref vecchia)
res = res[mask]
res.to_csv(sales_dest_path, index=False)
