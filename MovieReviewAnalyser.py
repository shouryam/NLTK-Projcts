#Library that allows us to read csv files
import pandas as pd
#Other Libraries needed
import numpy as np
#Allows us to remove all non letters
import re
#Allows us to remove html makrup from reviews in the data
from bs4 import BeautifulSoup
#allows us to remove stopwords
from nltk.corpus import stopwords
#allows us to simplify words down to roots "Stem the words"
from nltk.stem.porter import PorterStemmer
#Allows us to create a vector for each review
from sklearn.feature_extraction.text import CountVectorizer
#the Machine learning part that predicts the sentiment of the reviews
from sklearn.ensemble import RandomForestClassifier
#reads the csv file using pandas and fills it in the data frame called train
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# a function that simplifies each review 
def Review_Simplifier (review):
    #removs html markup with Beautiful soup
    review_noHTML = BeautifulSoup(review,"lxml").get_text()
    #replaces everything that is not a letter with a space
    letters_only = re.sub("[^a-zA-Z]"," ",review_noHTML)
    #makes all letters lowercase 
    lower_case = letters_only.lower()
    #makes a list with all all the lowercase words
    words = lower_case.split()
    #Gets rid of all the stopwords
    set_stopwords = stopwords.words("english")
    words = [w for w in words if not w in set_stopwords]
    #Joins the words toghether putting a space in between them
    " ".join(words)
    #stems the words into a base form and makes a list of them
    porter_stemmer = PorterStemmer()
    for w in range(len(words)):
        words[w] = porter_stemmer.stem(words[w])
    #join them back toghether 
    return(" ".join(words))
#gets the number of reviews for this case, 25,000
num_reviews = train["review"].size

#gives updates every 1000 reviews the program goes through
print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []

for i in range( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews ) )                                                                   
    #for each review it cleans it up and then adds it to the list called clean_train_reviews
    clean_train_reviews.append( Review_Simplifier( train["review"][i] ))
#Creates a bag of words for each review, here it specifies the details about the bag  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
#Creates the bag of words itslef
train_data_features = vectorizer.fit_transform(clean_train_reviews)
#makes the bag of words an array
train_data_features = train_data_features.toarray()

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 
#Fits the trees to the data so the trees can predict the test data
forest = forest.fit( train_data_features, train["sentiment"] )

#pandas reads the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

#Prints the shape of the test data
print (test.shape)

#get the length of the test data and create an array for the simplified test data
num_reviews = len(test["review"])
clean_test_reviews = [] 

#gives progress updates on how many test data reviews have been gone through
print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    #Adds the clean reviews to a list 
    clean_review = Review_Simplifier( test["review"][i] )
    clean_test_reviews.append( clean_review )
#creates a new bag of words for the test data and makes it an array 
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#Uses the random forest to predict the sentiment for the test data
result = forest.predict(test_data_features)
#Creates a datframe of the prediction with the sentiment coming from the previous line
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
#outputs it to csv file so we can see how good the prediciton was
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
