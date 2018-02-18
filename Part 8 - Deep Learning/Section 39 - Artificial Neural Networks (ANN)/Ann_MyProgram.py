#Artificial Neural Networks

#Installing Theano








# Part 1 - Data Preprocessing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #Creating Dummy Variables
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #Excludes first dummy variable to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling #Compulsory for ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN!

#Importing the keras library
import keras #Uses tensorflow in backend
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier= Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 )) 
#output_dim represents the no of hidden layers (which is generally the average of the input layers and output layers [(11+1)/2)] here, o/p layer is taken as 1 as the dataset has a binary dependent variable
#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) #input_dim need not be specified as the hidden layer knows what input node to expect 

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#adam is a type of stochastic gradient descent algorithm
#Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100  )
#nb_epoch represents how many times the rounds need to be compiled

# Part - 3 - making the predictions and evaluating the model

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #to convert y_pred from probability to True-False values

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

