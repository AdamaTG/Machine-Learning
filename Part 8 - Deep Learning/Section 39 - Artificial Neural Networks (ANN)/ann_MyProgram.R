#ANN



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

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1) #Allows us to connect to a specific server #nthreads allow us to specify the no of cores to use (-1 selects an optimal value)
classifier = h2o.deeplearning(y = 'Exited',  #To specify the dependent variable
                              training_frame = as.h2o(training_set), #converts the dataset to a h2o frameset
                              activation = 'Rectifier',
                              hidden = c(6,6), #c(6,6) represents no of layers in 1st and 2nd hidden layer
                              epochs = 100, #No of times the ANN algo runs over the dataset 
                              train_samples_per_iteration = -2) #-2 indicates a suitable batch size

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

