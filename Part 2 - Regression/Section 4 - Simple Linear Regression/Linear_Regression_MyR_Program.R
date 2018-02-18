# Data Preprocessing

# Importing the Dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

#splitting Dataset into Training Set and Test Set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3]) #we use [2:3] as 1 and 4 are not originally numeric
# test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting Simple Linear Regression to the Training Set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

#Predicting the Test Set Results
y_pred = predict(regressor, newdata = test_set)

#Visualizing the Training Set Results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

#Visualizing the Test set results
library(ggplot2)
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')






















