import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\insurance_data.csv')

X = df[['age']]
y = df.bought_insurance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test, y_test))
