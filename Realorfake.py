# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:30:34 2020

@author: Aakash Babu
"""
# to remove the warning in my code
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os
os.chdir("D:\\Studies\\Machine Learning\\Real or Not NLP with Disaster Tweets")
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train_data = pd.read_csv("D:\\Studies\\Machine Learning\\Real or Not NLP with Disaster Tweets\\data\\train.csv")

test_data = pd.read_csv("D:\\Studies\\Machine Learning\\Real or Not NLP with Disaster Tweets\\data\\test.csv")

check=pd.read_csv("D:\\Studies\\Machine Learning\\Real or Not NLP with Disaster Tweets\\data\\sample_submission.csv")


x_train = train_data.iloc[:,3] 
y_train = train_data["target"]
x_test  = test_data.iloc[:,3] 
y_test = check["target"]



#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=[1,0])


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()

nb.fit(tfidf_train,y_train)

y_pred1=nb.predict(tfidf_test.toarray())
score1=accuracy_score(y_test,y_pred1)
print(f'Accuracy: {round(score1*100,2)}%')

print(y_pred)
