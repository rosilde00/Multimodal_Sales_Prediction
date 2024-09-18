import pandas as pd

data = pd.read_csv('C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\aggregated_sales_month.csv')


aggregated_data = data.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode','DepartmentCode','WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription', 'Month','IdProdotto'], as_index=False)['Quantity'].sum()
aggregated_data.to_csv('C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\nonegozio_month.csv', index=False)

data = data[~data['IdProdotto'].str.endswith(('_2.jpg', '_5.jpg'))]
data = data.sort_values(['Descrizione', 'Month', 'LocationId'], ascending=[True, True, True])
data.to_csv('C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\front_img_month.csv', index=False)
