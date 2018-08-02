""" This machine takes in any sentence or paragraph provided by the user. Breaks into in tokens that can be proceesed by NLTK.
Removes the stop words (Words like of, a,...), provides the definition of all the important words
then counts and provides the frequency of all the words.
"""
#Import Libraries needed
import nltk
import numpy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
#Get library of english stopwords 
from nltk.corpus import stopwords
stopwords.words('english')
#Take User Input for paragraph
sentence = str(input("Enter the paragraph or sentence you want broken down: \n"))
#Break down string into tokens that NLTK can process. Fill a list with the tokens
tokens_in_sentance = nltk.word_tokenize(sentence)
#Fill a new list with the tokens 
important_words = tokens_in_sentance[:] 
#Go through the list of tokens and remove any stopwords
sr = stopwords.words('english')
for token in tokens_in_sentance:
    if token in stopwords.words('english'):
        important_words.remove(token)
#Go though the new list of only non-stop words and print out the definition
y = len(important_words)
for x in range(y):
    print("Definition of " + important_words[x] +  " ")
    syn = wordnet.synsets(important_words[x])
    #If the list syn has no words in it then that means wordnet does not have the definition of the word
    if len(syn) ==  0:
        print("No Definition for this word")
    else:
        print(syn[0].definition())
#Count the frequency of all the words and print it out
freq = nltk.FreqDist(important_words) 
print("Frequency of important words")
for key,val in freq.items(): 
    print (str(key) + ':' + str(val))
