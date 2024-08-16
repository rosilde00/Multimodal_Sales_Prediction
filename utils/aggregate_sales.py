import pandas as pd

#crea i dati aggregati
path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\sales_anagrafica_final.csv'
dest_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\aggregated_sales_season.csv'

data = pd.read_csv(path, sep=',')
data_aggregate = data.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode','DepartmentCode','WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription','LocationId','IdProdotto'], as_index=False)['Quantity'].sum() #aggrega mensilmente
data_aggregate.to_csv(dest_path, index=False)