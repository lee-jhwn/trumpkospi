import pandas as pd

kospi_data = pd.read_csv('kospi_17011908.csv')
kospi_data = kospi_data.dropna()
kospi_data['percent'] = 100*(kospi_data['kospi'] / kospi_data['kospi'].shift(1) - 1).fillna(method='bfill')

# print(kospi_data)


