import pandas as pd

tweets = pd.read_csv('tweets_prep_sentiment.csv', sep=',',parse_dates=['created_at'])

print(tweets)