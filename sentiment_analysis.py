from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd

"""sentiment analysis on the tweets using the RoBERTa model"""

#load csv into pandas data frame
tweets = pd.read_csv('tweets_prep_wo_rt.csv', sep=',',parse_dates=['created_at'])

#list of the text of each tweet
list_of_tweets = []
for i in range(len(tweets)):
    list_of_tweets.append(tweets['full_text'][i])

#preprocess tweets by changing usernames to @user
proc_tweets = []
for tweet in list_of_tweets:
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)
    proc_tweets.append(tweet_proc)

# load model and tokenizer
roberta = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

#lists to cache the results of the model 
list_of_neg = []
list_of_neu = []
list_of_pos = []

#use the RoBERTa model on each tweet to calculate its sentiment
counter = 0
for tweet in proc_tweets:
    # sentiment analysis
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #create the general sentiment which only classifies the tweet as neutral, if its probability is over 70%
    list_of_neg.append(scores[0])
    list_of_neu.append(scores[1])
    list_of_pos.append(scores[2])
    counter += 1
    print(counter)

#add sentiment to new created columns and save as csv
for i in range(len(tweets)):
    tweets.loc[i, 'neg'] = list_of_neg[i]
    tweets.loc[i, 'neu'] = list_of_neu[i]
    tweets.loc[i, 'pos'] = list_of_pos[i]

tweets['sentiment'] = tweets['pos'] - tweets['neg']

tweets.to_csv('tweets_prep_sentiment_wo_rt.csv', encoding='utf-8',index=False, sep=',')