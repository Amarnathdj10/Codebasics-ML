import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target
#print(df)
df['digit'] = df['target'].apply(lambda x : digits.target_names[x])


X = df.drop(['target','digit'], axis='columns')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = SVC()
model.fit(X_train, y_train)
print(model.predict([digits.data[67]]))
print(model.score(X_test,y_test))