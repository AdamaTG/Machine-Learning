#K means clustering

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing model dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values #The company is only interested in the relationship b/w Income and Spending score, not other metrics like age and gender

#Using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11): #To use the elbow method, we need to compute wcss for 10 iterations, we use 11 as upper bound is excluded
    kmeans  = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) #init means initialising the clusters, we use k-means++ to avoid random initialization trap
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_) #Wcss is also called inertia hence, we are appending the wcss values to our wcss array we created in line 14
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the null dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) #No of optimal clusters is 5
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans ==0,1], s= 100, c='red', label = 'Cluster1 - Careful') #s = size, X[y_kmeans == 0, 0] represents the first cluster, here the 0 after the comma indicates first column in X (Annual income)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans ==1,1], s= 100, c='blue', label = 'Cluster2 - Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans ==2,1], s= 100, c='green', label = 'Cluster3 - Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans ==3,1], s= 100, c='cyan', label = 'Cluster4 - Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans ==4,1], s= 100, c='magenta', label = 'Cluster5 - Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()