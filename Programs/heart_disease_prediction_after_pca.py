import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\CSV files\heart.csv')

cols_to_one_hot_encode = ['Sex','ChestPainType','RestingBP','RestingECG','ExerciseAngina','ST_Slope']
df = pd.get_dummies(df,columns=cols_to_one_hot_encode,drop_first=True)

X = df.drop(['HeartDisease'],axis=1)
y = df['HeartDisease']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca,y,test_size=0.2,random_state=42)
model.fit(X_train_pca,y_train)
print(model.score(X_test_pca,y_test))