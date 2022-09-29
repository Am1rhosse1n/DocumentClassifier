from __future__ import unicode_literals
from hazm import *
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
import sys
import codecs
import os
import os.path
#nltk.download('stopwords')
#nltk.download('punkt')
normalizer = Normalizer()
for i in range(23):        
    classes = os.listdir("./ohsumed-all/C"+str(i+1))   
    counter = 1
    print("Process Class " + str(i+1) + ": " )
    for data in classes :
        #print(str(counter) + ' of ' + str(len(classes)))
        counter = counter + 1
        data = "./ohsumed-all/C"+str(i+1) + "/" + data
        with codecs.open(data,'r',"utf-8") as f:
            Normalize = normalizer.normalize(f.read())
            stop_words = set(stopwords.words('english'))  
            word_tokens = word_tokenize(Normalize)    
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
print("____________")
print("Loading..")
