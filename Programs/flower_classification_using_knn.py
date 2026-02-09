import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x : iris.target_names[x])

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

'''plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='blue')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='green')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()'''

X = df.drop(['target','flower_name'],  axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))