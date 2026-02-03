import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\carprices2.csv')

X = df[['mileage','age']]
y = df['sell_price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)
print(model.predict(X_test))

print(model.score(X_test, y_test))
