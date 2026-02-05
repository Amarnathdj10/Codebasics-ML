import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits = load_digits()
#print(dir(digits))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))
print(model.predict([digits.data[67]]))

y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred,y_test)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()