import pandas as pd

tweets = pd.read_csv('tweets_prep_sentiment.csv', sep=',',parse_dates=['created_at'])

tweets['sentiment'] = tweets['pos'] - tweets['neg']

tweets.to_csv('tweets_prep2.csv', encoding='utf-8',index=False, sep=',')