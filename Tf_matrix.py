from __future__ import unicode_literals
#from hazm import *
from nltk import *
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import codecs
import os
import os.path
from nltk.stem import PorterStemmer

def create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    freq_table = {}
    words = word_tokenize(sentences)
    for word in words:
        word = word.lower()
        word = ps.stem(word)
        if word in stopWords:
            continue

        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    frequency_matrix[sentences[:15]] = freq_table

    return frequency_matrix


def create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_doc = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_doc

        tf_matrix[sent] = tf_table

    return tf_matrix



path = "./C1"
token_dict = {}

classes = os.listdir(path)   
print("Process Class 1: ")
for data in classes :
    data = "./C1/" + data
    print("Filename: " + data)
    with open(data) as f:
        text = f.read()
        frequency_matrix = create_frequency_matrix(text)
        tf_matrix = create_tf_matrix(frequency_matrix)
        token_dict[data] = tf_matrix


