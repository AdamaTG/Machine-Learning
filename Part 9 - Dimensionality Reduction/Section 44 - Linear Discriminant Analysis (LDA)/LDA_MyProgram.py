#LDA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values #We are selecting only age and salary as independent variables
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling (Should be used for PCA and LDA)
from sklearn.preprocessing import StandardScaler #To get more accurate results
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2) #We use None here initially instead of 2 since we need to compare all the variances of the independent variables and then choose the two best ones.
X_train = lda.fit_transform(X_train, y_train) #We take the dependent variable (y_train in this case) as LDA is supervised
X_test = lda.transform(X_test)

# Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state  = 0) #To get the same result
classifier.fit(X_train,y_train)

#Prediciting Test set Results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix #confusion_matrix is a function, not class (Hence no Capital letters)
cm = confusion_matrix(y_test, y_pred) #First variable is the correct values and the second variable is the ones predicted 
 
#Visualising the Training Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue'))) #3 colors here
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend()
plt.show()

#Visualising the Test Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), #We are plotting the axis for ages here (We use min()-1 and max()+1 to ensure the points are not on the axis)
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) #We are plotting the axis for salaries here
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), #contour is for making the contour line that separates the 0 and 1 states (red and green planes)
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue'))) #If pixel belongs to class 0, it is colorized in red, else green
plt.xlim(X1.min(), X1.max()) #plotting the points of the age
plt.ylim(X2.min(), X2.max()) #plotting the points of the salary
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], #scatter represents plotting of the points 
                c = ListedColormap(('red', 'green','blue'))(i), label = j) #colour coding the points 
plt.title('Logistic Regression (Test set)')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend()
plt.show()