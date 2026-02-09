import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\CSV files\titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1,inplace=True)
target = df.Survived
inputs = df.drop(['Survived'],axis=1)
dummies = pd.get_dummies(inputs.Sex)
inputs = pd.concat([inputs,dummies],axis=1)
inputs.drop(['Sex'],axis=1,inplace=True)
inputs.Age.fillna(math.floor(inputs.Age.mean()),inplace=True)

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model = GaussianNB()

model.fit(X_train,y_train)
print(model.score(X_test, y_test))