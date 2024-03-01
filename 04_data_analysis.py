import pandas as pd

#load csvs into pandas dataframes
tweets = pd.read_csv('tweets_labeled.csv', sep=',',parse_dates=['created_at'])

#add sentiment groups to entries
for i,row in tweets.iterrows():
    if row["sentiment"] < -0.7:
        tweets.loc[i,'sentiment_group'] = -0.7
    elif row["sentiment"] >= -0.7 and row["sentiment"] < -0.3:
        tweets.loc[i,'sentiment_group'] = -0.3
    elif row["sentiment"] > -0.3 and row["sentiment"] < 0.3:
        tweets.loc[i,'sentiment_group'] = 0
    elif row["sentiment"] > 0.3 and row["sentiment"] <= 0.7:
        tweets.loc[i,'sentiment_group'] = 0.3
    else:
        tweets.loc[i,'sentiment_group'] = 0.7

#create sub-dataset for each location based on coordinates
loc0 = tweets[tweets['kmeans_label'] == 0]
loc0['k_means_location'] = loc0.user_location.mode()[0]
loc1 = tweets[tweets['kmeans_label'] == 1]
loc1['k_means_location'] = loc1.user_location.mode()[0]
loc2 = tweets[tweets['kmeans_label'] == 2]
loc2['k_means_location'] = loc2.user_location.mode()[0]
loc3 = tweets[tweets['kmeans_label'] == 3]
loc3['k_means_location'] = loc3.user_location.mode()[0]
loc4 = tweets[tweets['kmeans_label'] == 4]
loc4['k_means_location'] = loc4.user_location.mode()[0]
loc5 = tweets[tweets['kmeans_label'] == 5]
loc5['k_means_location'] = loc5.user_location.mode()[0]
loc6 = tweets[tweets['kmeans_label'] == 6]
loc6['k_means_location'] = loc6.user_location.mode()[0]

print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc0.iloc[0,20]) +": " +str(len(loc0[(loc0['sentiment'] < -0.7) & (loc0['retweet_count'] > 10)])/len(loc0)))
print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc1.iloc[0,20]) +": " +str(len(loc1[(loc1['sentiment'] < -0.7) & (loc1['retweet_count'] > 10)])/len(loc1)))
print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc2.iloc[0,20]) +": " +str(len(loc2[(loc2['sentiment'] < -0.7) & (loc2['retweet_count'] > 10)])/len(loc2)))
print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc3.iloc[0,20]) +": " +str(len(loc3[(loc3['sentiment'] < -0.7) & (loc3['retweet_count'] > 10)])/len(loc3)))
print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc4.iloc[0,20]) +": " +str(len(loc4[(loc4['sentiment'] < -0.7) & (loc4['retweet_count'] > 10)])/len(loc4)))
print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc5.iloc[0,20]) +": " +str(len(loc5[(loc5['sentiment'] < -0.7) & (loc5['retweet_count'] > 10)])/len(loc5)))
print("Percentage of negative (<-0.7) Tweets with more than 10 Retweets in " + str(loc6.iloc[0,20]) +": " +str(len(loc6[(loc6['sentiment'] < -0.7) & (loc6['retweet_count'] > 10)])/len(loc6)))