#Upper Confidence Bound

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB (No package available)
import math
N = 10000 #No of rounds
d = 10 #No of arms
ads_selected  = [] #To find the add with max UCB
numbers_of_selections = [0] * d #[0]*d represents the fact that no ads have been clicked initially #the number of times each ad was selected upto round n (In this case for 10,000 users)
sums_of_rewards = [0] * d  #This represents the fact that the sum of the rewards is initially 0
total_reward = 0
for n in range(0, N):#Looping over 10,000 users
    ad = 0
    max_upper_bound = 0
    for i in range(0, d): #looping over the 10 different ads
        if(numbers_of_selections[i]>0): #To apply the following only if at least 1 ad is selected
            average_reward = sums_of_rewards[i]/ numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/ numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound #To select the max_upper_bound
            ad = i #to create index of max_upper_bound
    ads_selected.append(ad) #To select the required
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 # to show the updated ad 
    reward  = dataset.values[n, ad ] #To get the real reward
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
#Visualizing the results
    plt.hist(ads_selected)
    plt.title('Histogram of ad selections')
    plt.xlabel('Ads ')
    plt.ylabel('Number of times each ad was selected')
    plt.show()
