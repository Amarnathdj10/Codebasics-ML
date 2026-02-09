import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

wine = load_wine()
#print(dir(wine))
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = wine.target

scaler = MinMaxScaler()
df['proline'] = scaler.fit_transform(df[['proline']])

X = df.drop(['target'],axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf1 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('nb',  MultinomialNB())
])

clf2 = Pipeline([
    ('scaler', MinMaxScaler()), 
    ('nb', GaussianNB())
])

clf1.fit(X_train, y_train)
print(clf1.score(X_test,y_test))
clf2.fit(X_train, y_train)
print(clf2.score(X_test,y_test))