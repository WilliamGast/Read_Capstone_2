from typing import Text
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#from __future__ import division
from collections import Counter
import pandas as pd
import numpy as np
import string
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#dense_features=train_features.toarray()
#dense_test= X_test.toarray()
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, VectorizerMixin
from sklearn.model_selection import GridSearchCV
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Capstone_2_code import cleanTXT

def test_model(Xtrain,Xtest, ytrain, ytest,Vectorizer,Model):
    """
    This is the function that will be testing and recieving the report for whatever model you want.
    ...
    Parameters
    ----------
    Xtrain,Xtest,ytrain,ytest : 
        these are the inputst that you will recieve from yourr train_test_split
    Vectorizer:
        This is where you will input the vecorizer
    Model:
        This is Where you will unput the model that you want to test
    """
    model = make_pipeline((Vectorizer(stop_words = stopwords)), Model())
    model.fit(Xtrain, ytrain)
    label = model.predict(Xtest)
    metrics = classification_report(label,ytest)
    return metrics

def gridsearch(Xtrain,ytrain):
    """
    This function will preform gridsearch for hyperparameter testing 
    ...
    Parameters
    ----------
    Xtrain,Xtest,ytrain,ytest : 
        these are the inputst that you will recieve from yourr train_test_split
    Vec_and_mod:
        This is where you will input a list expressing the model and vector that you want to use
    """
    '''

    '''
    parameters = {
        #'class_prior':
        #'fit_prior': (True,False),
        'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
        }
    '''
    Class  = Pipeline([
        ('vec', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),]) 
    '''
    model = MultinomialNB()   
    gs = GridSearchCV(model,parameters,cv=2,n_jobs = -1,verbose = 2)
    gs = gs.fit(Xtrain,ytrain)
    bestscore =  gs.best_score_
    for param_name in sorted(parameters.keys()):
        return "%s: %r" % (param_name, gs.best_params_[param_name]) 


if __name__ == "__main__":
    data = pd.read_csv('../data/Tweets.csv')
    df = data.copy()[['airline_sentiment', 'text']]
    df['text'] = df['text'].apply(cleanTXT)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df['text'], df['airline_sentiment'], test_size=0.2, random_state=0,stratify=df['airline_sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2, random_state=0)
    
    
    stopwords = set(stopwords.words('english'))

    metrics = test_model(X_train,X_test, y_train, y_test,CountVectorizer,MultinomialNB)
    #look at the difference between CountVectorizer and Tfidf
    Tfidf_metrics = test_model(X_train,X_test, y_train, y_test,CountVectorizer,MultinomialNB)
    #print(metrics)
    print(Tfidf_metrics)
    Class  = Pipeline([
        ('vec', CountVectorizer()),
        #('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),])
    #pipeline = Pipeline([('vec', CountVectorizer( tokenizer=CountVectorizer(), stop_words=stopwords, token_pattern=None)),('tfidf', TfidfTransformer()), ('clf',MultinomialNB())])
    #gs = gridsearch(X_train,y_train)
    #print(gs)
    #print(MultinomialNB().get_params().keys())
    print(df.head())
