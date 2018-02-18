#Apriori

#Data Preprocessing
install.packages('arules')
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
library(arules)
dataset1 = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) #sep = ',' indicates that the variables are separated by commas in the dataset
#rm.duplicates removes any duplicate values in the same row which may have occured due to human error
summary(dataset1)
itemFrequencyPlot(dataset1, topN = 10) #topN indicates most products purchasedd

#Training the Apriori model on the dataset
rules = apriori(data = dataset1, parameter = list(support = 0.004, confidence = 0.2 ))

#Visualizing the results
inspect(sort(rules, by = 'lift')[1:10]) #sorting the rules by top 10 lists