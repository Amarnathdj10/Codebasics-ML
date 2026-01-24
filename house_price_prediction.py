import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"D:\\Coding journey\\Codebasics ML\\homeprices.csv")

model = LinearRegression()
model.fit(df[['area']], df.price)
d = pd.read_csv(r'D:\\Coding journey\\Codebasics ML\\areas.csv')
p = model.predict(d[['area']])
d['prices'] = p
d.to_csv('prediction.csv', index=False)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,model.predict(df[['area']]),color='blue')
plt.show()