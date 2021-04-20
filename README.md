# Capstone-2

## Intruduction
 I plan to create a sentiment classification model with the sentiment classes being  negative,positive or neutral given tweets received by US airlines. As my baseline I will be using a lexicon and rule-based sentiment analysis called VADER and testing that against a Random forest and naive bayes to see which performs the best.

 Sentiment analysis or opinion mining is an NLP technique that lets you determine the attitude (positive, negative, or neutral) of text.

 With this model that I create from this project an airline company could take these results and understand how their customers are reacting to them and their services  from their twitter data. This is important because this can be used to improve a company's decision making, customer satisfaction and more
 
## Data
The data was a Kaggle data set named [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). 
![data info](./images/info.png)
![missing data](./images/missing.png)

### Data Cleaning
This data was already very clean for the columns that I ended up using. I did have to do some cleaning of @ signs and hashtags but other than that most of my data processing and feature enginearing was done in the preprocessing stage.


## EDA
![airline mood](./images/airline_mood.png)
![reasons](./images/reasons.png)
### Preprocessing
- Tokenization
- Lower casing
- Stop words removal
- Stemming
- Lemmatization

## Models
#### VADER sentiment 
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically tuned to be used for social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative. It also shows how positive or negative the text is.
#### Naive Bayes
Multnomial Naive Bayes classification is known for being a good classifier for sentiment analysis. The intuition behind Naive Bayes is to find the probability of classes assigned to given text by using the joint probabilities of words and classes. 

### Random Forest
The reason that I decided to try a random forest is because of its ability to handle large data sets with higher dimensionality and with more trees, it won't allow-overfitting trees in a model. 


### Choosing an evaluation metric
I prioritized using the F1-score for my evalusation metric. I did this because it gives a good measure of the incorrectly classified cases. F-1 score is used when the False Negatives and False Positives are the most important. It also does a good job of combatting class imbalance.

### Results
|               | Precision | Recall | f1-score |   |
|---------------|-----------|--------|----------|---|
| VADER         |    .70      |  .55     |    .58     |   |
|  Naive Bayes  |    .84      |  .77      |   .79       |   |
| Random Forest |    .77      |  .75      |   .76       |   |
### Example Output

### What Next?
- More Hyperparmeter Tuning 
- Try a CNN model