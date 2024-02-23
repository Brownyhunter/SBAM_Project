import pandas as pd

#load csvs into pandas dataframes
locations = pd.read_csv('location_geocode.csv',sep=",")
tweets = pd.read_csv('auspol2019.csv',sep=",",parse_dates=['created_at'])


#remove rows with wrong created_at value
tweets = tweets[tweets.created_at.str.contains("2019")]

#parse date strings to datetime format
tweets['created_at'] = pd.to_datetime(tweets.created_at)

#reduce data set from ~183000 to under 5000
#remove tweets that have not been retweeted
tweets_w_retweets = tweets.loc[tweets['retweet_count'] != 0]

#remove rows with missing values
tweets_w_retweets = tweets_w_retweets.dropna()

#add columns for latitude and longitude of location
locations = locations.rename(columns={'name': 'user_location'})
tweets_w_retweets_loc = pd.merge(tweets_w_retweets, locations, on=['user_location'], how='left')

#remove tweets from outside of australia based on geolocation (latitude [-44 - -10], longitude [110 - 155])
tweets_aus = tweets_w_retweets_loc.loc[(tweets_w_retweets_loc['lat'] > -44) & (tweets_w_retweets_loc['lat'] < -10) & (tweets_w_retweets_loc['long'] > 110) & (tweets_w_retweets_loc['long'] < 155)]

#take sample of the data to reduce the data set from ~40000 to 5000
tweets_sample = tweets_aus.sample(n=5000, random_state=2)

#reset the index of the data frame
tweets_sample = tweets_sample.reset_index(drop=True)

#save prepared data to csv
tweets_sample.to_csv('tweets_prep2.csv', encoding='utf-8',index=False, sep=',')