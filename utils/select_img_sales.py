import duckdb
import pandas as pd

aggregated_path = 'C:\\ORS\\Data\\aggregated_sales_final.xlsx'
data_path = 'C:\\ORS\\Data\\sales_anagrafica.xlsx'
dest_path = 'C:\\ORS\\Data\\sales_anagrafica_final.csv'

data = pd.read_excel(data_path)
aggregated = pd.read_excel(aggregated_path)
aggregated = aggregated.drop(columns=['qty'], axis='columns')

res = duckdb.query("SELECT * FROM data JOIN aggregated ON aggregated.prodcode LIKE REPLACE(data.idProdotto, '*', '%')").df()
res = res.drop(columns=['IdProdotto'], axis='columns')
res.to_csv(dest_path, index=False)