import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\CSV files\heart.csv')

cols_to_encode = ['Sex','ChestPainType','RestingBP','RestingECG','ExerciseAngina','ST_Slope']
df = pd.get_dummies(df,columns=cols_to_encode,drop_first=True)

X = df.drop('HeartDisease',axis=1)
y = df.HeartDisease

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

scores1 = cross_val_score(SVC(),X,y,cv=5)
print(scores1.mean())

bagging_model = BaggingClassifier(
    estimator=SVC(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=42
)


scores2 = cross_val_score(bagging_model,X,y,cv=5)
print(scores2.mean())

scores3 = cross_val_score(DecisionTreeClassifier(),X,y,cv=5)
print(scores3.mean())

bagging_model2 = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=42
)

scores4 = cross_val_score(bagging_model2,X,y,cv=5)
print(scores4.mean())