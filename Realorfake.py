# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:30:34 2020

@author: Aakash Babu
"""
import pandas as pd
import numpy as np
import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train = pd.read_csv("data/train.csv")

X = train.iloc[:,3]
y = train.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier()
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#DataFlair - Build confusion matrix
print(confusion_matrix(y_test,y_pred, labels=[1,0]))


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

nb.fit(tfidf_train,y_train)

y_pred1=nb.predict(tfidf_test.toarray())
score1=accuracy_score(y_test,y_pred1)
print(f'Accuracy: {round(score1*100,2)}%')

print(confusion_matrix(y_test,y_pred1, labels=[1,0]))

# saving vectorizer
with open('tfid.pickle','wb') as f:
    pickle.dump(tfidf_vectorizer,f)

# saving model
with open('model-faketweet.pickle','wb') as f:
    pickle.dump(nb,f)
