#NLP
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting = 3 ignores double quotes which might cause problems.

# Cleaning the texts
import re #re library is used for cleaning the review 
import nltk 
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus = [] #To put all the cleaned reviews in this list
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] ) # [^a-zA-Z] indicates what we should keep and it removes everything else
    # ' ' indicates we are replacing removed character with a space
    review = review.lower()               #For converting all characters to lower case
    review = review.split() #Splits the review into different words and converts it into a list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #set function increases the execution time
    review = ' '.join(review) #To convert the list back into a string separated by a space(indicated by ' '.)\
    corpus.append(review) #appending clean review to corpus
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max_features indicates we keep the 1500 most frequent words because not all irrelevant words would have been removed by stopwords
X = cv.fit_transform(corpus).toarray() #To create a sparse matrix of words, X is like an independent variable we created in Classification
# .toarray() is for converting the corpus into an array
y = dataset.iloc[:, 1].values #To store the dependant variable, ie the 'liked' variable, which are the predictions obtained after training the model

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Implementing Naive Bayes which is the most common algorithm used for NLP

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() #No parameters required
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(55+91)/200