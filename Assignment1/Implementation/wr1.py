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

def order(DataFrame):
	data = [[ float(DataFrame[0]),float(DataFrame[1]),float(DataFrame[2]),float(DataFrame[3]),float(DataFrame[4]),
			  float(DataFrame[5]),float(DataFrame[6]),float(DataFrame[7]),float(DataFrame[8])]]
	df = pd.DataFrame(data,columns=['MSSubClass','LotArea','OverallQual','LotFrontage',
									'OverallCond','BedroomAbvGr','TotRmsAbvGrd','TotalBsmtSF','YearBuilt'])
	return df

input_DataFrame = input()
input_DataFrame = input_DataFrame.split(',')

idf = order(input_DataFrame)
df = read_dataframe("houses.csv")

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
