from __future__ import unicode_literals
#from hazm import *
from nltk import *
from math import log, sqrt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import codecs
import os
import numpy as np
import os.path
from nltk.stem import PorterStemmer
import time
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

def create_tf_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    freq_table = {}
    words = word_tokenize(sentences)
    for word in words:
        if word.isalpha():
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
    
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

    frequency_matrix = freq_table
    return frequency_matrix

        
def tf_idf_matrix(freq_matrix):
    word_per_doc = {}
    doc = {}
    nt = {}
    doc_vector = {}
    doc_num = 0
    count = 0
    for docName in freq_matrix:
        doc_num = doc_num + 1
        doc = freq_matrix[docName]
        for word in doc:
            if word in word_per_doc:
                word_per_doc[word] += 1
            else:
                if (count<1024):    
                    word_per_doc[word] = 1
                    count = count + 1
        
        
    for word in word_per_doc:
        word_per_doc[word] = log(doc_num/word_per_doc[word])
 
    for i in freq_matrix:
        doc = freq_matrix[i]
        for j in word_per_doc:
            if j in doc:
                doc_vector[j] = word_per_doc[j]  * doc[j]
            else:
                doc_vector[j] = 0
        nt[i] = doc_vector
        doc_vector = {}

    return nt, word_per_doc

def cosine(nt):
    ntc = {}
    sum_of_square = 0
    for i in nt:
        doc = nt[i]
        for j in doc:
            sum_of_square = sum_of_square + (doc[j] ** 2)
        for k in doc:
            doc[k] = doc[k]/ sqrt(sum_of_square)
        ntc[i] = doc
        doc = {}
            
    return ntc


def PrepareData_for_SVM(ntc):    
    X = []
    X_list = []
    class_list = []

    for key in ntc:
        class_names = key[2:4]
        class_list.append(class_names)
        temp = ntc[key]
        for entity in temp:     
            value_temp = temp[entity]
            X.append(value_temp)
        X_list.append(X)
        X = []
    return X_list, class_list
########################################################################################

"""
# main:
"""

All_docs = []
All_tests = []
tf_dict = {}
ntc = {}
ntc_list = []

#Extract Train Data
for i in range(23):        
    classes = os.listdir("./C"+str(i+1))
    for j in classes:
        result = "./C"+str(i+1) + "/" + j
        All_docs.append(result)
    
start = time.time()
end = 0
#Processing Data
counter = 0
for data in All_docs :
    counter = counter+1
    progressing_index = counter/len(All_docs)
    print(str(int(progressing_index*100)) + str("% Processing Data" ) + "------  Time: " + str(int(end)))
    with open(data) as f:
        text = f.read()
        tf_matrix = create_tf_matrix(text)
        tf_dict[data] = tf_matrix
    end = time.time() - start
#Calculate ntc Matrix
print("Calculate ntc Matrix...")
nt, word_per_collection = tf_idf_matrix(tf_dict)
ntc = cosine(nt)

#### SVM ####
X, Y = PrepareData_for_SVM(ntc)
np.save('X.npy',X)
np.save('Y.npy',Y)
# dividing X, y into train and test data 



print("Done!")
