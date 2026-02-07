import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"D:\\Coding journey\\Codebasics ML\\CSV files\\canada_per_capita_income.csv")

model = LinearRegression()
model.fit(df[['year']],df.per_capita_income)

print(model.predict(pd.DataFrame([[2020]],columns=['year'])))

plt.scatter(df.year,df.per_capita_income,color='red',marker='+')
plt.plot(df.year,model.predict(df[['year']]),color='blue')
plt.xlabel('year')
plt.ylabel('per capita income')
plt.show()