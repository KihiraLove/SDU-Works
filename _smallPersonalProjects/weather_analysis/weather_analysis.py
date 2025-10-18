import pandas as pd
import datetime


def clean_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame.columns = ['time', 'mm']
    data_frame['month'] = data_frame['time'].apply(lambda x: str(x)[4:-2])
    data_frame = data_frame.query("month == \"08\" or month == \"09\"")
    del data_frame['month']

    data_frame['time'] = data_frame['time'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').date())

    return data_frame


budapest = pd.read_csv('data/precipitation/r_h_Budapest_19012023.csv', sep = ';', header=None)
debrecen = pd.read_csv('data/precipitation/r_h_Debrecen_19012023.csv', sep = ';', header=None)
keszthely = pd.read_csv('data/precipitation/r_h_Keszthely_19012023.csv', sep = ';', header=None)
miskolc = pd.read_csv('data/precipitation/r_h_Miskolc_19012023.csv', sep = ';', header=None)
nyiregyhaza = pd.read_csv('data/precipitation/r_h_Nyiregyhaza_19012023.csv', sep = ';', header=None)
pecs = pd.read_csv('data/precipitation/r_h_Pecs_19012023.csv', sep = ';', header=None)
sopron = pd.read_csv('data/precipitation/r_h_Sopron_19012023.csv', sep = ';', header=None)
szeged = pd.read_csv('data/precipitation/r_h_Szeged_19012023.csv', sep = ';', header=None)
szombathely = pd.read_csv('data/precipitation/r_h_Szombathely_19012023.csv', sep = ';', header=None)
turkeve = pd.read_csv('data/precipitation/r_h_Turkeve_19012023.csv', sep = ';', header=None)

rain_data = [clean_data(budapest),
             clean_data(debrecen),
             clean_data(keszthely),
             clean_data(miskolc),
             clean_data(nyiregyhaza),
             clean_data(pecs),
             clean_data(sopron),
             clean_data(szeged),
             clean_data(szombathely),
             clean_data(turkeve)]

print(rain_data)