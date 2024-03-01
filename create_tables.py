import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tweets = pd.read_csv('tweets_labeled.csv', sep=',',parse_dates=['created_at'])

low_sentiment = tweets[tweets['sentiment'] < -0.7]
print(low_sentiment.columns)
high_rt_low_sentiment = low_sentiment.nlargest(15,'retweet_count')[['retweet_count','full_text','sentiment', 'user_location']]
print(high_rt_low_sentiment)

high_rt_low_sentiment.to_excel('high_rt_low_sentiment.xlsx',index=False)