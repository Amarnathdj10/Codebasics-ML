import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Coding journey\Codebasics ML\CSV files\HR_comma_sep.csv")

left = df[df.left==1]
print(left.shape)
retained = df[df.left==0]
print(retained.shape)

df.groupby('left').mean(numeric_only=True)

pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.xlabel("Salary Level")
plt.ylabel("Employee Count")
plt.title("Employee Attrition by Salary")
plt.show()

pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.xlabel("Department Name")
plt.ylabel("Employee Count")
plt.title("Employee Attrition by Salary")
plt.show()

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

dummies = pd.get_dummies(subdf.salary, prefix='salary', drop_first=True)
df_with_dummies = pd.concat(
    [subdf,dummies],
    axis='columns'
)
df_with_dummies.drop('salary',inplace=True,axis='columns')

X = df_with_dummies.loc[:, df_with_dummies.columns!='salary']
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test, y_test))