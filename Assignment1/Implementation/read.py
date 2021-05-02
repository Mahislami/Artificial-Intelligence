import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def solve_equation(a,b,c,e,f,g,dataframe):
	pass

df = pd.read_csv("houses.csv")
#Drop the categorical datas.
df = df.drop(columns=['LotConfig','Neighborhood'])
df = df.fillna(df.mean())
SalePrice = df['SalePrice']
Id = df['Id']
#print(df)
df.plot.scatter(x='MSSubClass', y='SalePrice',c='Blue',title='Year Built')
df.plot.scatter(x='LotArea', y='SalePrice',c='Blue',title='Lot Area')
df.plot.scatter(x='OverallQual', y='SalePrice',c='Blue',title='Overall Qual')
df.plot.scatter(x='LotFrontage', y='SalePrice',c='Blue',title='Lot Frontage')
df.plot.scatter(x='OverallCond', y='SalePrice',c='Blue',title='Overall Cond')
df.plot.scatter(x='BedroomAbvGr', y='SalePrice',c='Blue',title='Bedroom AbvGr')
df.plot.scatter(x='TotRmsAbvGrd', y='SalePrice',c='Blue',title='Tot RmsAbvGrd')
df.plot.scatter(x='TotalBsmtSF', y='SalePrice',c='Blue',title='Total BsmtSF')
df.plot.scatter(x='YearBuilt', y='SalePrice',c='Blue',title='Year Built')

#plt.show()

#Eq1 => Aw + Bb = C
A = np.sum(np.square(df['OverallQual']))
B = np.sum(df['OverallQual'])
C = np.sum(df['OverallQual']*df['SalePrice'])
#Eq2 => Ew + Fb = G
E = np.sum(df['OverallQual'])
F = 1133
G = np.sum(df['SalePrice'])

Eq1 = np.array([[A,B], [E,F]])
Eq2 = np.array([C,G])
Ans = np.linalg.solve(Eq1, Eq2)
print(Ans)

x = df['OverallQual']
y = Ans[0]*x + Ans[1]

df.plot.scatter(x='OverallQual', y='SalePrice',c='Blue',title='Overall Qual')
plt.plot(x, y, '-r')
plt.grid()
plt.show()

Rmse = np.sqrt(np.square(np.subtract(Ans[0]*df['OverallQual'] + Ans[1],df['SalePrice'])).mean())
print(Rmse)

