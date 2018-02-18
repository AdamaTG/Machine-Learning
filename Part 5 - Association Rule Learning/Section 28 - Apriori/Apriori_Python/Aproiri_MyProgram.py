#Apriori

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501): #We are looping over the no of transactions to ensure that the dataset is in proper format
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)]) #This converts the dataset into a list of products to be accepted by the algorithm

#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2 , min_lift = 3, min_length = 2) 
# min_support = 3*7/7500 -> We are considering a product purchased an average of 3 times a day over a week over 7500 datasets
#min_confidence value is a convenient value found over many trials
#min_lift value is also a tried and tested value
#min_length is the minimum no of products in our rules

#Visualizing the results
results = list(rules)

# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])