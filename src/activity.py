from Capstone_2_code import cleanTXT
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt

if __name__ == "__main__":
    
    data = pd.read_csv('../data/Tweets.csv')
    data['text'] = data['text'].apply(cleanTXT)
    #print(data.head())
    data['tweet_created'] = data['tweet_created'].astype('datetime64[ns]')
    data['tweeted_at_year_month'] = data['tweet_created'].dt.strftime('%Y-%m-%d')
    months = data['tweeted_at_year_month']
    data = data.sort_values(by='tweeted_at_year_month')
    negative = data[data['airline_sentiment'] == 'negative'].groupby(['tweeted_at_year_month']).sum()
    positive = data[data['airline_sentiment'] == 'positive'].groupby(['tweeted_at_year_month']).sum()
    neutral = data[data['airline_sentiment'] == 'neutral'].groupby(['tweeted_at_year_month']).sum()

    plt.plot(data['tweeted_at_year_month'].unique(), positive, color='green')
    plt.plot(data['tweeted_at_year_month'].unique(), negative, color='red')
    plt.plot(data['tweeted_at_year_month'].unique(), neutral, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Activity')
    plt.title('Activity of tweets')
    plt.show()
    
    
    
    #plt.show()
    