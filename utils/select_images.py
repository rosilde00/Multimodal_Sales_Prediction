import pandas as pd

path = 'C:\\ORS\\Data\\sales_anagrafica.xlsx' 
dest_path = 'C:\\ORS\\Data\\products_id.xlsx'

data = pd.read_excel(path)
ids = data[['IdProdotto']].drop_duplicates()
ids.to_excel(dest_path, index=False)