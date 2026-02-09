import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv(r'D:\Coding journey\Codebasics ML\CSV files\spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train.values)

model = MultinomialNB()
model.fit(X_train_vectorized,y_train)

email = ['Upto 20% discount on parking, get your offers now!']

email_vectorized = vectorizer.transform(email)
print(model.predict(email_vectorized))