import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print([str(iris.target_names[i]) for i in y_pred])

