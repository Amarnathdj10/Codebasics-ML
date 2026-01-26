import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from word2number import w2n

df = pd.read_csv("D:\Coding journey\Codebasics ML\hiring.csv")

df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)

median_experience = math.floor(df.experience.median())
df.experience = df.experience.fillna(median_experience)
median_test_score = math.floor(df.test_score.median())
df.test_score = df.test_score.fillna(median_test_score)
print(df)


model = LinearRegression()
model.fit(df[['experience','test_score','interview_score']],df.salary)

print(model.predict([[2,9,6]]))
print(model.predict([[12,10,10]]))