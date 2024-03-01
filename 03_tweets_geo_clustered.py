import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#load csvs into pandas dataframes
tweets = pd.read_csv('tweets_prep_sentiment.csv', sep=',',parse_dates=['created_at'])

#decrease to relevant columns
x=tweets.loc[:,['id','lat','long']]

#form clusters
K_clusters = range(1,20)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = x[['lat']]
X_axis = x[['long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# Visualize elbow curve
plt.plot(K_clusters, score)
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

kmeans = KMeans(n_clusters = 7, init ='k-means++', random_state=1)
kmeans.fit(x[x.columns[1:3]]) # Compute k-means clustering.
x['kmeans_label'] = kmeans.fit_predict(x[x.columns[1:3]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(x[x.columns[1:3]]) # Labels of each point

#visualize 'map of australia'
x.plot.scatter(x = 'long', y = 'lat', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=200, alpha=0.5)
plt.show()

#array of distances of each entry to each centroid
x_dist = kmeans.transform(x[x.columns[1:3]])

#Add centroids
x = x[['id','kmeans_label']]
clustered_data = tweets.merge(x, left_on='id', right_on='id')
print(centers)

u=1
for i in centers:
    z = pd.DataFrame({"id": u,"lat": [i[0]], "long": [i[1]], "full_text": ['Centroid ' + str(u)]})
    clustered_data = pd.concat([clustered_data, z], ignore_index=True)
    u += 1

clustered_data['dist_to_centroid'] = np.nan
t = 0
for i in clustered_data:
    clustered_data.at[t,'dist_to_centroid'] = min(x_dist[t])
    t += 1

#centroid for sentiments
clustered_data = pd.concat([clustered_data, pd.DataFrame({"id": [10],"sentiment": [-1], 'full_text': ['Negative']})], ignore_index=True)
clustered_data = pd.concat([clustered_data, pd.DataFrame({"id": [11],"sentiment": [-0.7], 'full_text': ['Semi negative']})], ignore_index=True)
clustered_data = pd.concat([clustered_data, pd.DataFrame({"id": [12],"sentiment": [0], 'full_text': ['Neutral']})], ignore_index=True)
clustered_data = pd.concat([clustered_data, pd.DataFrame({"id": [13],"sentiment": [0.7], 'full_text': ['Semi positive']})], ignore_index=True)
clustered_data = pd.concat([clustered_data, pd.DataFrame({"id": [14],"sentiment": [1], 'full_text': ['Positive']})], ignore_index=True)

print(clustered_data.iloc[0])

#create edge table for sentiments
edges = pd.DataFrame(columns=['Source', 'Target', 'Type'])
for i,row in clustered_data.iterrows():
    if row["sentiment"] < -0.7:
        target = 10
    elif row["sentiment"] >= -0.7 and row["sentiment"] < -0.3:
        target = 11
    elif row["sentiment"] > -0.3 and row["sentiment"] < 0.3:
        target = 12
    elif row["sentiment"] > 0.3 and row["sentiment"] <= 0.7:
        target = 13
    elif row["sentiment"] > 0.7:
        target = 14
    z = pd.DataFrame({'Source': [row['id']], 'Target': [target], 'Type': ['Undirected']})
    edges = pd.concat([edges, z], ignore_index=True)


edges.to_csv('tweets_labeled_edges_sentiment.csv', encoding='utf-8',index=False, sep=',')

#create edge table for geolocation
geo_edges = pd.DataFrame(columns=['Source', 'Target', 'Type', 'Weight'])
for i,row in clustered_data.iterrows():
    if i < 5000:
        target = x_dist[i].argmin() + 1
        z = pd.DataFrame({'Source': [row['id']], 'Target': [target], 'Type': ['Undirected'], 'Weight': [1/min(x_dist[i])]})
        geo_edges = pd.concat([geo_edges, z], ignore_index=True)

geo_edges.to_csv('tweets_labeled_edges_geo.csv', encoding='utf-8',index=False, sep=',')

clustered_data.to_csv('tweets_labeled.csv', encoding='utf-8',index=False, sep=',')