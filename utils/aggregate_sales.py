import pandas as pd

path = 'C:\\ORS\\Data\\sales.csv'
dest_path = 'C:\\ORS\\Data\\aggregated_sales.xlsx'
data = pd.read_csv(path, names=['locid', 'prodcode', 'colorid', 'year', 'week', 'qty', 'netvalue'], header=None, sep=';')
data = data.drop(columns=['locid', 'year', 'week', 'netvalue'], axis=1)

for idx in range(0, len(data.values)):
    ref = data.iloc[idx].values[0]
    ref = ref[:3] + '_' + ref[4:7] + '_' + ref[7:12] + '_' + ref[12:14] + '_' + data.iloc[idx].values[1] + '_*.jpg'
    data.iloc[idx,0] = ref

data = data.drop(columns=['colorid'])  
data_aggregate = data.groupby('prodcode', as_index=False).sum()
data_aggregate.to_excel(dest_path, index=False)