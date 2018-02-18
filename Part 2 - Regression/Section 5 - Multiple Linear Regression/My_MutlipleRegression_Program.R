# Importing the Dataset

#Importing the Dataset
dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

#splitting Dataset into Training Set and Test Set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3]) #we use [2:3] as 1 and 4 are not originally numeric
# test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting Multiple Linear Regression to the Training Set
regressor = lm(formula = Profit ~ .,
               data = training_set)

#Predicting the Test Set Results
y_pred = predict(regressor, newdata = test_set)

#Building optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset) #We can use Training Set as well
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset) #We can use Training Set as well
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset) #We can use Training Set as well
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset) #We can use Training Set as well
summary(regressor)



