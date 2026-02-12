import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\CSV files\diabetes.csv')
X = df.drop('Outcome',axis=1)
y = df.Outcome

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

scores = cross_val_score(DecisionTreeClassifier(),X,y,cv=5)
decision_tree_score = scores.mean()
print(decision_tree_score)

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=42
)
scores2 = cross_val_score(bag_model,X,y,cv=5)
bagging_model_score = scores2.mean()
print(bagging_model_score)
