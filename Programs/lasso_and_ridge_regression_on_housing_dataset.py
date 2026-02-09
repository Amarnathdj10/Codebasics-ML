import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\CSV files\Melbourne_housing_FULL.csv')

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
              'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
df = df[cols_to_use]
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].mean())
df = df.dropna()

df = pd.get_dummies(df, drop_first=True)
X = df.drop(['Price'],axis=1)
y = df['Price']

model_parameters = {
    'Linear Regression': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('linear', LinearRegression())
        ]),
        'params': {}
    },
    'Lasso Regression': {
        'model': Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
    ]),
        'params': {
            'lasso__alpha': [0.1, 1, 10]
        }
    },
    'Ridge Regression': {
        'model' : Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(max_iter=10000))
    ]),
        'params': {
            'ridge__alpha':[0.1,1,10]
        }
    }
}

scores = []
for model_name, mp in model_parameters.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False,verbose=True)
    clf.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_parameters': clf.best_params_
    })

df = pd.DataFrame(scores,columns=['model','best_score','best_parameters'])
print(df)