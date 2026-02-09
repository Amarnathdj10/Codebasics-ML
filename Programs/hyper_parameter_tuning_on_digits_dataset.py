from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()

model_parameters = {
    'SVM': {
        'model': SVC(gamma='auto'),
        'params': {
            'C': [1,10,20],
            'kernel':['rbf','linear']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1,5,10]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,10,20]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'params': {}
    },
    'MultinomialNB': {
        'model': MultinomialNB(),
        'params': {}
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
        }
    }
}

scores = []
for model_name, mp in model_parameters.items():
    clf = GridSearchCV(mp['model'],mp['params'], cv=5, return_train_score=False)
    clf.fit(digits.data,digits.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_parameters': clf.best_params_
    })

df = pd.DataFrame(scores,columns=['model','best_score','best_parameters'])
print(df)