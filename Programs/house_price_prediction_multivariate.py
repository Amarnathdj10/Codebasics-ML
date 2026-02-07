import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\Coding journey\Codebasics ML\CSV files\homeprices2.csv")
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

model = LinearRegression()
model.fit(df[['area','bedrooms','age']], df.price)

print(model.coef_)
print(model.intercept_)

print(model.predict([[3000,3,40]]))