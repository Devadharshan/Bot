import nltk
import numpy as np

#nltk.download('punkt') #if we are running nltk for first time

from nltk.stem.porter import PorterStemmer  # for the stemming we can use many thing we are using porter stemmer

stemmer = PorterStemmer()



def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words),dtype=np.float32) #np.zeros used to fill in with 0
    for idx ,w in enumerate(all_words): # enumerate to have like this [(index),word]
        if w in tokenized_sentence:  # if word is in the tokenized word
            bag[idx]= 1.0  # assigning that to 1
        return bag







