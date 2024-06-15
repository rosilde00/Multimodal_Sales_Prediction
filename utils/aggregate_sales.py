import pandas as pd

path = 'C:\\ORS\\Data\\sales.csv'
dest_path = 'C:\\ORS\\Data\\aggregated_sales.xlsx'
data = pd.read_csv(path, names=['locid', 'prodcode', 'colorid', 'year', 'week', 'qty', 'netvalue'], header=None, sep=';')
data = data.drop(columns=['locid', 'year', 'week', 'netvalue'], axis=1)

data_aggregate = data.groupby(['prodcode', 'colorid'], as_index=False).sum()


for idx in range(0, len(data_aggregate.values)):
    ref = data_aggregate.iloc[idx].values[0]
    ref = ref[:3] + '_' + ref[4:7] + '_' + ref[7:12] + '_' + ref[12:14] + '_' + data_aggregate.iloc[idx].values[1] + '_*.jpg'
    data_aggregate.iloc[idx,0] = ref

data_aggregate = data_aggregate.drop(columns=['colorid'], axis='columns')
data_aggregate.to_excel(dest_path, index=False)