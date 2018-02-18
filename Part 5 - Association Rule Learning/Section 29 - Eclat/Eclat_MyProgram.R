#Eclat

#Data Preprocessing
install.packages('arules')
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
library(arules)
dataset1 = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) #We are creating a Sparse Matrix here #sep = ',' indicates that the variables are separated by commas in the dataset
#rm.duplicates removes any duplicate values in the same row which may have occured due to human error
summary(dataset1)
itemFrequencyPlot(dataset1, topN = 10) #topN indicates most products purchasedd

#Training the Eclat model on the dataset
rules = eclat(data = dataset1, parameter = list(support = 0.004, minlen = 2))

#Visualizing the results
inspect(sort(rules, by = 'support')[1:10]) #sorting the rules by top 10 lists