import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def solve_equation(A,B,C,E,F,G):
	#Eq1 => Aw + Bb = C
	Eq1 = np.array([[A,B], [E,F]])
	Eq2 = np.array([C,G])
	Ans = np.linalg.solve(Eq1, Eq2)
	return Ans
	
	
df = pd.read_csv("houses.csv")

#Drop the categorical datas.
df = df.drop(columns=['LotConfig','Neighborhood'])
df = df.fillna(df.mean())


df.plot.scatter(x='MSSubClass', y='SalePrice',c='Blue',title='Year Built')
df.plot.scatter(x='LotArea', y='SalePrice',c='Blue',title='Lot Area')
df.plot.scatter(x='OverallQual', y='SalePrice',c='Blue',title='Overall Qual')
df.plot.scatter(x='LotFrontage', y='SalePrice',c='Blue',title='Lot Frontage')
df.plot.scatter(x='OverallCond', y='SalePrice',c='Blue',title='Overall Cond')
df.plot.scatter(x='BedroomAbvGr', y='SalePrice',c='Blue',title='Bedroom AbvGr')
df.plot.scatter(x='TotRmsAbvGrd', y='SalePrice',c='Blue',title='Tot RmsAbvGrd')
df.plot.scatter(x='TotalBsmtSF', y='SalePrice',c='Blue',title='Total BsmtSF')
df.plot.scatter(x='YearBuilt', y='SalePrice',c='Blue',title='Year Built')


Ans = solve_equation(np.sum(np.square(df['OverallQual'])), np.sum(df['OverallQual']),
					 np.sum(df['OverallQual']*df['SalePrice']), np.sum(df['OverallQual']),
					 1133, np.sum(df['SalePrice']))

print(Ans)

x = df['OverallQual']
y = Ans[0]*x + Ans[1]

df.plot.scatter(x='OverallQual', y='SalePrice',c='Blue',title='Overall Qual')
plt.plot(x, y, '-r')
plt.grid()
plt.show()

Rmse = np.sqrt(np.square(np.subtract(Ans[0]*df['OverallQual'] + Ans[1],df['SalePrice'])).mean())
print(Rmse)

