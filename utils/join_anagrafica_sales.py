import pandas as pd
#fa la join fra anagrafica e sales settimanali
def img_ref(season, catr, ccol):
    new_ref = list()
    for sn, ca, cc in zip(season, catr, ccol):
        sn, ca, cc = str(sn), str(ca), str(cc)
        r = sn + '_' + ca[:3] + '_' + ca[3:8] + '_' + ca[8:] + '_' + cc + '_*.jpg'
        new_ref.append(r)
    return new_ref

def join_ref(season, catr):
    new_ref = list()
    for sn, ca in zip(season, catr):
        sn, ca = str(sn), str(ca)
        r = sn + '-' + ca
        new_ref.append(r)
    return new_ref

path_anagrafica = 'C:\\ORS\\Data\\Anagrafica.xlsx' 
path_sales = 'C:\\ORS\\Data\\sales.csv'
path_dest = 'C:\\ORS\\Data\\sales_anagrafica.xlsx'

anagrafica = pd.read_excel(path_anagrafica)
sales = pd.read_csv(path_sales, names=['LocationId', 'ProductCode', 'CodiceColore', 'Year', 'Week', 'Quantity', 'NetValue'], header=None, sep=';')

ref_images = img_ref(anagrafica['Stagione'], anagrafica['CodiceArticolo'], anagrafica['CodiceColore']) #crea la reference in formato nome immagine
ref_join = join_ref(anagrafica['Stagione'], anagrafica['CodiceArticolo']) #modifica l'id per la join
anagrafica = anagrafica.drop(['Stagione', 'CodiceArticolo', 'DescrizioneColore', 'AreaDescription', 
                      'CategoryDescription', 'SectorDescription', 'DepartmentDescription', 'WaveDescription',
                      'AstronomicalSeasonDescription', 'SalesSeasonBeginDate', 'SalesSeasonEndDate'], axis='columns')

anagrafica['ProductCode'] = ref_join
anagrafica['IdProdotto'] = ref_images #sarà l'id prodotto che rimane

sales = sales.drop(['NetValue', 'Year'], axis='columns')
new_dataset = anagrafica.merge(sales, on=['ProductCode', 'CodiceColore'], how='inner')
new_dataset = new_dataset.drop(['ProductCode'], axis='columns')
new_dataset.to_excel(path_dest, index=False)
