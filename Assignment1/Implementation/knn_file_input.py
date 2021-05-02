import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_dataframe(filename):
	idf = pd.read_csv(filename)
	idf = idf.fillna(idf.mean())
	return idf

def standardalize_dataframe(df):
	sdf = df - df.min()
	sdf = sdf / (sdf.max() - sdf.min())
	return sdf


idf = read_dataframe("input.csv")
df = read_dataframe("houses.csv")
df = df.drop(columns=['LotConfig','Neighborhood'])
SalePrice = df['SalePrice']
Id = df['Id']

sdf = standardalize_dataframe(df)
sdf = sdf.drop(columns=['Id','SalePrice'])
ddf = df.drop(columns=['Id','SalePrice'])


idf = idf - ddf.min()
idf = idf / (ddf.max() - ddf.min())

distance = np.square(np.subtract(sdf, idf.iloc[0]))
Ans = distance.sum(axis=1)
Ans = np.sqrt(Ans)

Min = idx = np.argpartition(Ans, 10)
Ans1 = SalePrice[idx[:10]]
avg = Ans1.mean()
print(avg)
