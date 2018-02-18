# K-Means Clustering

#Importing mall dataset
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[4:5]

#using elbow method to find optimal number of clusters
set.seed(6)
wcss  = vector() #initialising an empty cluster
for (i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = "b", main = paste('Clusters of clients'), xlab = "Number of clusters", ylab = "WCSS")
  
#Applying k-means to the model dataset with 5 clusters
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

#Visualizing the clusters
library(cluster)
clusplot(X, 
         kmeans$cluster, #To indicate that the cluster belongs to the kmeans variable
         lines = 0,      #We give lines = 0 which means we don't get lines betwen distances
         shade = TRUE,   #To shade clusters W.R.T their density
         color = TRUE,   
         labels = 2,      #To label all the 200 points W.R.T their corresponding cluster
         plotchar = FALSE, 
         span = TRUE,     #To plot ellipses around our clusters
         main = paste('Clusters of clients'),
         xlab = "Annual Income",
         ylab = "Spending Score")