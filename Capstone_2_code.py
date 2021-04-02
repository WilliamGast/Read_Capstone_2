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

def cleanTXT(text):
    """
    This function is going to clean your text columns from any
    account tags (ex:@VirginAirways), Hashtags, and Retweet tags)
    ...
    Parameters
    ----------
    text : str
        Columns where your tweets are stored
    """
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    return text

def analyize_text(input_text,analyzer):
    """
    This function is where we are running are sentiemnt with VADER sentiment and TextBlob
    We will be using the VADER sentiment as our baseline
    ...
    Parameters
    ----------
    input_text : str
        Columns where your tweets are stored
    analyzer : 
        This is where you will pass in the analyzer that you want to run (either VADER or TextBlob)
    """
    if analyzer == VADER:
        result_ = analyzer.polarity_scores(input_text)
        score = result_['compound']
    elif analyzer == Textblob:
        score = TextBlob(input_text).sentiment.polarity
    if score == 0:
        result = 'neutral'
    if score > 0:
        result = 'positive'
    if score < 0:
        result = 'negative'
    return result

def matrix_and_array(target_column,columns):
    """
    This function will create your multiclass confusion matrix for the VADER adn TextBlob

    ...
    Parameters
    ----------
    target_column : 
        Column where you original sentiment analysis is stored
    columns:
        this is where you will input the model columns of tested sentiment (VADER and/or TextBlob column)
    """
    matrix_list = []
    for i in columns:
        create_matrix = confusion_matrix(df[target_column], df[i])
        create_vector = np.array(create_matrix)
        create_vector = np.reshape(create_vector,-1)
        matrix = list(create_vector)
        matrix_list.append(matrix)
    return matrix_list


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

def gridsearch(Xtrain,Xtest, ytrain, ytest,Vec_and_mod):
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
    Vec_and_mod.fit(Xtrain,ytrain)
    predicted = Class.predict(Xtest)
    metrics = classification_report(predicted,ytest)
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }    
    gs = GridSearchCV(Class,parameters,cv=5,n_jobs = -1)
    gs = gs.fit(X_train,y_train)
    bestscore =  gs.best_score_
    for param_name in sorted(parameters.keys()):
        return "%s: %r" % (param_name, gs.best_params_[param_name])
if __name__ == "__main__":
    # Read in tweet data
    data = pd.read_csv('Tweets.csv')
    df = data.copy()[['airline_sentiment', 'text']]
    df['text'] = df['text'].apply(cleanTXT)
    stopwords = set(stopwords.words('english'))
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df['text'], df['airline_sentiment'], test_size=0.2, random_state=0,stratify=df['airline_sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2, random_state=0)
    VADER =  SentimentIntensityAnalyzer()
    Textblob = TextBlob
    df['vader'] = df['text'].apply(analyize_text, analyzer = VADER)
    df['Textblob'] = df['text'].apply(analyize_text, analyzer = Textblob)

    col_names = ['TN','FP','FN','TP']
    columns =  ['vader','Textblob']
    index = ['vader','TextBlob']
    matrix_list = matrix_and_array('airline_sentiment', columns)
    #truth = pd.DataFrame(matrix_list,index = index,columns = col_names).sort_values('TN', ascending = False)
    #print(confusion_matrix(df['airline_sentiment'], df['vader']).ravel())
    #print(classification_report(df['airline_sentiment'],df['vader']))
    #print(classification_report(df['airline_sentiment'],df['Textblob']))
    print(test_model(X_train, X_test, y_train, y_test,CountVectorizer,RandomForestClassifier))
    print(test_model(X_train, X_test, y_train, y_test,CountVectorizer,MultinomialNB))
    Class  = text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),])
    '''
    Class.fit(X_train,y_train)
    predicted = Class.predict(X_test)
    metrics = classification_report(predicted,y_test)
    print(metrics)
    '''
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    
   
    
    
