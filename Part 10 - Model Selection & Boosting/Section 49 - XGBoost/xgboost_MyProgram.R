#XGBoost

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

#Encoding categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting xgboost to the Training Set
#install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10) #We need matrix type for xgboost to work

#Applying xgboost
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(X){ #function(X) creates a variable X containing the dataset stored in folds
  training_fold = training_set[-X,] #X represents test set, hence we are excluding it
  test_fold = training_set[X, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2])/(cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
}) #applying the cross validation
accuracy1 = mean(as.numeric(cv))
