import pandas as pd

#crea i dati aggregati
path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\sales_anagrafica_final.csv'
dest_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\aggregated_sales_month.csv'

data = pd.read_csv(path, sep=',')
data['month'] = (data['Week'] - 1) // 4
data_aggregate = data.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode','DepartmentCode','WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription','LocationId','IdProdotto', 'month'], as_index=False)['Quantity'].sum() #aggrega mensilmente
data_aggregate.drop(['month'], axis='columns')
data_aggregate.to_csv(dest_path, index=False)