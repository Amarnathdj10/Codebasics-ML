import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv(r"D:\Coding journey\Codebasics ML\CSV files\titanic.csv")

inputs = df[['Pclass','Sex','Age','Fare']]
target = df.Survived

inputs.Sex = inputs.Sex.map({'female':0,'male':1})
inputs.Sex = inputs.Sex.fillna(inputs.Sex.mode()[0])
age_median = math.floor(inputs.Age.median())
inputs.Age = inputs.Age.fillna(age_median)

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model = DecisionTreeClassifier(max_depth=3,random_state=42)

model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))

plt.figure(figsize=(10,7))
plot_tree(
    model,
    feature_names=inputs.columns,
    class_names=['Not survived', 'Survived'],
    filled = True
)
plt.show()