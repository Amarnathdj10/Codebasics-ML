import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_digits

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier()
svm = SVC(max_iter=500)

def get_score(model, X_train, X_test, y_train, y_test): 
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

folds = StratifiedKFold(n_splits=3)

scores_lr = []
scores_rf = []
scores_svm = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                        digits.target[train_index], digits.target[test_index]
    scores_lr.append(get_score(lr,X_train,X_test,y_train,y_test))
    scores_rf.append(get_score(rf,X_train,X_test,y_train,y_test))
    scores_svm.append(get_score(svm,X_train,X_test,y_train,y_test))

print(scores_lr)
print(scores_rf)
print(scores_svm)