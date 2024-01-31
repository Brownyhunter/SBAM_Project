import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

tweets = pd.read_csv('tweets_prep2.csv', sep=',',parse_dates=['created_at'])

#decrease to relevant columns
x=tweets.loc[:,['id','lat','long']]

#form clusters
K_clusters = range(1,20)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = x[['lat']]
X_axis = x[['long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# # Visualize
# plt.plot(K_clusters, score)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show()

kmeans = KMeans(n_clusters = 7, init ='k-means++', random_state=1)
kmeans.fit(x[x.columns[1:3]]) # Compute k-means clustering.
x['kmeans_label'] = kmeans.fit_predict(x[x.columns[1:3]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(x[x.columns[1:3]]) # Labels of each point

#visualize
x.plot.scatter(x = 'long', y = 'lat', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=200, alpha=0.5)
plt.show()

x_dist = kmeans.transform(x[x.columns[1:3]])
print(x_dist[0])

x = x[['id','kmeans_label']]
clustered_data = tweets.merge(x, left_on='id', right_on='id')
#print(clustered_data.head())