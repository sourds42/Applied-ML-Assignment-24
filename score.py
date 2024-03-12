import joblib
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
tfidf_transformer = TfidfTransformer()
vectorizer = CountVectorizer()
filename = r'C:\Users\ASUS\Documents\Assignment_2\Assignment 2\best_model.joblib'
best_model = joblib.load(filename)
train_data = pd.read_csv("train.csv")
X_train = train_data['processed_text']
X_train_counts = vectorizer.fit_transform(X_train)
X_train_vec = tfidf_transformer.fit_transform(X_train_counts)

import numpy as np

def score(text:str, model, threshold):
    # Convert text to a numpy array
    k = {'message': [text]}
    testing_data = pd.DataFrame(k)
    X_test = testing_data['message']
    X_test_counts = vectorizer.transform(X_test)
    X_test_vec = tfidf_transformer.transform(X_test_counts)


    # Predict using the model
    propensity = ((model.predict_proba(X_test_vec)).tolist())[0][1]
    if propensity >= threshold:
        prediction = 1
    else:
        prediction = 0
    return prediction, propensity
