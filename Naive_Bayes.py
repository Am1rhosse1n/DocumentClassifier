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

    
    
def create_Naive_matrix(tf_dict):
    language_model = []
    recorde = {}
    for class_name in range(4):
        for i in tf_dict:
            if (i[3] == str(class_name + 1)):
                recorde[i] = tf_dict[i]
                
        language_model.append(recorde)
        recorde = {}
    
    
    word_per_doc = {}
    doc = {}
    doc_num = 0
    languageModel = []
    for class_name in language_model:
        test = class_name
        for docName in test:
            doc_num = doc_num + 1
            doc = test[docName]
            for word in doc:
                if word in word_per_doc:
                    word_per_doc[word] += doc[word]
                else:
                    word_per_doc[word] = doc[word]
        languageModel.append(word_per_doc)
        word_per_doc = {}

    
    language_model = []
    naive = {}
    for dic in languageModel:
        test = dic
        dic_lenght = len(dic)
        for word in dic:
           naive[word] = dic[word] / dic_lenght
        language_model.append(naive)
        naive = {}

    return language_model



def create_Naive_matrix_test(tf_dict):
    naive_matrix_test = {}
    for i in tf_dict:
        query = tf_dict[i]
        for word in query:
            query[word] = query[word]/ len(query)
        naive_matrix_test[i] = query
    return naive_matrix_test
            
    language_model = []
    recorde = {}
    for class_name in range(4):
        for i in tf_dict:
            if (i[3] == str(class_name + 1)):
                recorde[i] = tf_dict[i]
                
        language_model.append(recorde)
        recorde = {}
    return language_model


def naive_result(All_tests,naive_dict):
    tf_dict_test = {}
    counter = 0
    for data in All_tests :
        counter = counter+1
        progressing_index = counter/len(All_tests)
        print(str(int(progressing_index*100)) + str("% Processing Test Data" ))
        with open(data) as f:
            text = f.read()
            tf_matrix_test = create_tf_matrix(text)
            tf_dict_test[data[5:20]] = tf_matrix_test
             
    naive_test_dict = create_Naive_matrix_test(tf_dict_test) 
    temp = 0
    semi_result = []
    naive_result = []
    for query_index in naive_test_dict:
        query = naive_test_dict[query_index]
        for class_name in naive_dict:
            for word in query:
                class_dict = class_name
                if word in class_dict:
                    temp = temp + class_dict[word]
                else:
                    continue
            semi_result.append(temp)
            temp = 0        
        semi_result = np.argmax(semi_result) + 1
        naive_result.append(semi_result)
        semi_result = []
    
    naive_GT = []
    for index in naive_test_dict:
        naive_GT.append(int(index[3]))
     
    true = 0
    for i in range(len(naive_result)):
        if naive_result[i] == naive_GT[i]:
            true = true + 1
    accuracy = true/len(naive_result) * 100
    return accuracy
    
    
    
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
for i in range(4):        
    classes = os.listdir("./C"+str(i+1))
    for j in classes:
        result = "./C"+str(i+1) + "/" + j
        All_docs.append(result)
    
#Extract Test Data
for ii in range(4):
    classes = os.listdir("./test/C" + str(ii+1))
    for j in classes:
        result = "./test/C" + str(ii+1) + "/" + j
        All_tests.append(result)


#Processing Data
counter = 0
for data in All_docs :
    counter = counter+1
    progressing_index = counter/len(All_docs)
    print(str(int(progressing_index*100)) + str("% Processing Data" ))
    with open(data) as f:
        text = f.read()
        tf_matrix = create_tf_matrix(text)
        tf_dict[data] = tf_matrix


#Naive Result
naive_dict = create_Naive_matrix(tf_dict)
accuracy = naive_result(All_tests,naive_dict)


print("Done!")
