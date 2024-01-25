import pandas as pd

tweets = pd.read_csv('tweets_prep_sentiment.csv', sep=',',parse_dates=['created_at'])

tweets['sentiment'] = tweets['pos'] - tweets['neg']

tweets['neg_sentiment'] = tweets['sentiment'] < -0.7
tweets['pos_sentiment'] = tweets['sentiment'] > 0.7

tweets_edges = tweets['neg_sentiment', 'pos_sentiment']
print(tweets_edges)



#tweets.to_csv('tweets_prep2.csv', encoding='utf-8',index=False, sep=',')