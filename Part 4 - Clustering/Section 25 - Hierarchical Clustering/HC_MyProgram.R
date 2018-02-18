#Hierarchical Clustering

# Importing the Mall_Customers dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#Using the dendogram to find the optimal number of clusters
dendrogram = hclust(dist(X, method = 'euclidean'),  method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogran'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

#Fitting hierarchical clustering to the mall dataset
hc = hclust(dist(X, method = 'euclidean'),  method = 'ward.D')
y_hc = cutree(hc, 5)

#Visualizing the clusters
library(cluster)
clusplot(X, 
         y_hc, #To indicate that the cluster belongs to the kmeans variable
         lines = 0,      #We give lines = 0 which means we don't get lines betwen distances
         shade = TRUE,   #To shade clusters W.R.T their density
         color = TRUE,   
         labels = 2,      #To label all the 200 points W.R.T their corresponding cluster
         plotchar = FALSE, 
         span = TRUE,     #To plot ellipses around our clusters
         main = paste('Clusters of clients'),
         xlab = "Annual Income",
         ylab = "Spending Score")