#Upper Confidence Bound

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling (No package available)
import random
N = 10000 #No of rounds
d = 10 #No of arms
ads_selected  = [] 
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):#Looping over 10,000 users
    ad = 0
    max_random = 0
    for i in range(0, d): #looping over the 10 different ads
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta #To select the max_upper_bound
            ad = i #to create index of max_upper_bound
    ads_selected.append(ad) #To select the required
    reward  = dataset.values[n, ad ] #To get the real reward
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
         numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
#Visualizing the results
    plt.hist(ads_selected)
    plt.title('Histogram of ad selections')
    plt.xlabel('Ads ')
    plt.ylabel('Number of times each ad was selected')
    plt.show()
