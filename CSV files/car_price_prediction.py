import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\carprices.csv')

X = df[['car_model','mileage','age']]
y = df['sell_price']

ct = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(sparse_output=False, drop='first'),['car_model'])
    ],
    remainder='passthrough'
)

X_encoded = ct.fit_transform(X)

model = LinearRegression()
model.fit(X_encoded,y)

test = pd.DataFrame({
    'car_model': ['Mercedez Benz C class'],
    'mileage': [45000],
    'age': [4]
})

test_encoded = ct.transform(test)
print(model.predict(test_encoded))

print(model.score(X_encoded,y))