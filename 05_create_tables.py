import pandas as pd

#load csvs into pandas dataframes
tweets = pd.read_csv('tweets_labeled.csv', sep=',',parse_dates=['created_at'])

#create excel files with top 50 retweet count tweets with negative sentiment
low_sentiment = tweets[tweets['sentiment'] < -0.7]
high_rt_low_sentiment = low_sentiment.nlargest(50,'retweet_count')[['retweet_count','full_text','sentiment', 'user_location']]

high_rt_low_sentiment.to_excel('high_rt_low_sentiment3.xlsx', index=False)