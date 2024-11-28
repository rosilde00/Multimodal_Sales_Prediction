import pandas as pd

path = '...'
dest_folder = '...'

data = pd.read_csv(path, sep=',')


#aggrega sui negozi, mantenendo la granularita' settimanale
data_aggregate = data.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode','DepartmentCode',
                               'WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription','LocationId','IdProdotto', 'Week'], 
                              as_index=False)['Quantity'].sum() 
data_aggregate.to_csv(f'{dest_folder}\\week_nonegozi.csv', index=False)


#calcola il mese della settimana corrispondente (ad esempio, la settimana 3 appartiene al mese 1, la settimana 4 al mese 2)
data['Month'] = ((data['Week'] - 1) // 4) + 1
data_aggregate['Month'] = ((data['Week'] - 1) // 4) + 1

#dati mensili non aggregati sui negozi
data_month = data.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode','DepartmentCode',
                           'WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription','LocationId','IdProdotto', 'Month'], 
                          as_index=False)['Quantity'].sum()
data_month.to_csv(f'{dest_folder}\\data_month.csv', index=False)

#dati mensili aggregati sui negozi
data_month_aggr = data_aggregate.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode',
                                          'DepartmentCode','WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription',
                                          'IdProdotto', 'Month'], as_index=False)['Quantity'].sum()
data_month_aggr.to_csv(f'{dest_folder}\\month_nonegozi.csv', index=False)

#dati stagionali non aggregati sui negozi
data_season = data.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode','DepartmentCode',
                           'WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription','LocationId','IdProdotto'], 
                          as_index=False)['Quantity'].sum()
data_season.to_csv(f'{dest_folder}\\data_season.csv', index=False)

#dati stagionali aggregati sui negozi
data_season_aggr = data_aggregate.groupby(['Descrizione','CodiceColore','PianoTaglia','AreaCode','CategoryCode','SectorCode',
                                           'DepartmentCode','WaveCode','AstronomicalSeasonExternalID','SalesSeasonDescription',
                                           'IdProdotto'], as_index=False)['Quantity'].sum()
data_season.to_csv(f'{dest_folder}\\season_nonegozi.csv', index=False)
