#NLP
#Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
dataset = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)#To ignore the quotes, stringsAsFactors = FALSE - to prevent identifying the parameters as False
#The original term here is because the dataset will be modified later and so we need to keep an original version of it
# Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers) #To remove unwanted numbers
corpus = tm_map(corpus, removePunctuation) #to remove punctuation
corpus = tm_map(corpus, removeWords, stopwords()) # To remove irrlevant words (the, is ,an etc)
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace) #To remove any unwanted spaces that could have arisen beause of the previous steps

#Creating the Bag of words model
dtm = DocumentTermMatrix(corpus) #Creates a sparse matrix
dtm = removeSparseTerms(dtm, 0.999) #99.9% of most frequent words will be kept, one-off words appearing in few reviews will be removed
dataset = as.data.frame(as.matrix(dtm)) #To convert the dtm of type sparse matrix into a data frame to ensure compatibilty with rest of the code 
dataset$Liked = dataset_original$Liked  #We are appending the dependant variable Liked to the Independent variables

dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random forest classifier to the Training set
#install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          n_tree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)