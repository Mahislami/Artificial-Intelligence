import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_dataframe(filename):
	idf = pd.read_csv(filename)
	idf = idf.drop(columns=['LotConfig','Neighborhood'])
	idf = idf.fillna(idf.mean())
	return idf

def standardalize_dataframe(df):
	sdf = df - df.min()
	sdf = sdf / (sdf.max() - sdf.min())
	return sdf

data = [70,11435,8,67.66037735849056,7,3,7,792,1929]
idf = pd.DataFrame([data],columns=['MSSubClass', 'LotArea', 'OverallQual', 'LotFrontage', 'OverallCond', 
           'BedroomAbvGr', 'TotRmsAbvGrd', 'TotalBsmtSF', 'YearBuilt'] )

df = read_dataframe("houses.csv")

SalePrice = df['SalePrice']
Id = df['Id']

sdf = standardalize_dataframe(df)
sdf = sdf.drop(columns=['Id','SalePrice'])
ddf = df.drop(columns=['Id','SalePrice'])

idf = idf - ddf.min()
idf = idf / (ddf.max() - ddf.min())

def predict(dataframe):
	distance = np.square(np.subtract(sdf, dataframe.iloc[0]))
	Ans = distance.sum(axis=1)
	Ans = np.sqrt(Ans)
	Min = idx = np.argpartition(Ans, 10)
	Ans1 = SalePrice[idx[:10]]
	avg = Ans1.mean()
	return avg

Price = predict(idf)
print(Price)

