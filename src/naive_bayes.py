import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk
import itertools
import matplotlib.pyplot as plt
from joblib import dump, load
import os
from Capstone_2_code import cleanTXT
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
def lem_tok(text):

    lemmatizer = WordNetLemmatizer()
    words = text.split()

    # additional lemmatization terms
    additional_lemmatize_dict = {
        "cancelled": "cancel",
        "cancellation": "cancel",
        "cancellations": "cancel",
        "delays": "delay",
        "delayed": "delay",
        "baggage": "bag",
        "bags": "bag",
        "luggage": "bag",
        "dms": "dm",
        "thanks": "thank"
    }
    
    tokens = []
    for word in words:
        if word not in stopword:
            if word in additional_lemmatize_dict:
                clean_word = additional_lemmatize_dict[word]
            else:
                clean_word = lemmatizer.lemmatize(word)
            tokens.append(clean_word)
    return tokens

if __name__=='__main__':
    data = pd.read_csv('../data/Tweets.csv')
    data['text'] = data['text'].apply(cleanTXT)
    

    X = data['text']
    y = data['airline_sentiment']

    # split into train/test
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data['text'], data['airline_sentiment'], test_size=0.2, random_state=0,stratify=data['airline_sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2, random_state=0)
    
    #lemmatizer_ = WordNetLemmatizer()
    stop = stopwords.words('english')
    stopword = stop + ['united','usairways','americanair','jetblue','southwestair','http','still','us','virginamerica','co','flightled']
    count_vect = CountVectorizer(
                        tokenizer=lem_tok,
                        analyzer='word',
                        stop_words=stopword,
                        max_features=1000
                    )
                    
    nb_model = MultinomialNB()

    tfidf_transformer = TfidfTransformer(use_idf=True)

    nb_pipeline = Pipeline([
                            ('vect', count_vect),
                            ('tfidf', tfidf_transformer),
                            ('model', nb_model),
                            ])
    
    nb_pipeline.fit(X_train, y_train)
    
    y_preds = nb_pipeline.predict(X_test)
    
    

    target_names = np.unique(y)
    n = 15  # number of top words to include
    feature_words = count_vect.get_feature_names()
    for cat in range(len(np.unique(y))):
        print(f"\nTarget: {cat}, name: {target_names[cat]}")
        log_probs = nb_model.feature_log_prob_[cat]
        top_idx = np.argsort(log_probs)[::-1][:n]
        features_top_n = [feature_words[i] for i in top_idx]
        print(f"Top {n} tokens: ", features_top_n)
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
