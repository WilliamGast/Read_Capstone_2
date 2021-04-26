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
    stopword = stop + ['united','usairways','americanair','jetblue','southwestair','http','still','us','virginamerica','co','flightled','!','“','im','guy','got','thanks!','you!','get']
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
    n = 10  # number of top words to include
    feature_words = count_vect.get_feature_names()
    for i in range(len(np.unique(y))):
        print(f"\nTarget: {i}, name: {target_names[i]}")
        log_probs = nb_model.feature_log_prob_[i]
        top_idx = np.argsort(log_probs)[::-1][:n]
        features_top_n = [feature_words[i] for i in top_idx]
        print(f"Top {n} tokens: ", features_top_n)
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        ax.set_title(f'Top {n} Feature importance for {target_names[i]} sentiment', fontsize=18)
        ax.set_ylabel('log probability ', fontsize=22)
        ax.set_xlabel('features', fontsize=22)
        ax.tick_params(labelsize=18)
                
        feat_importances = np.array(nb_model.feature_log_prob_[i])
        feat_names = np.array(feature_words)
        sort_idx = feat_importances.argsort()[::-1][:n]
        feat_std_deviations = []
        if len(feat_std_deviations) > 0:
            feat_std_deviations = feat_std_deviations[sort_idx]
        else:
            feat_std_deviations = None
        ax.bar(feat_names[sort_idx], feat_importances[sort_idx],
             linewidth=1, yerr=feat_std_deviations)
        ax.set_xticklabels(feat_names[sort_idx], rotation=40, ha='right')
        plt.tight_layout()
        #plt.savefig(outfilename)
        #plt.close('all')

        plt.show()
        
    Mood_count=data['negativereason'].value_counts()
    Index = [1,2,3,4,5,6,7,8,9,10]
    plt.bar(Index,Mood_count)
    plt.xticks(Index,['Customer Service issues','Late Flight','Can not Tell','Cancelled Flight','Lost Luggage','Bad Flight',' Flight Booking Problem','Flight attendent compaint',' long lines','Damaged luggage' ],rotation=80,fontsize=16)
    plt.ylabel('Count',fontsize=22)
    plt.xlabel('Reason',fontsize=22)
    plt.title('Reason for Negative Sentiment',fontsize=22)
    plt.show()