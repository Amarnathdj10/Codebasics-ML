from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
