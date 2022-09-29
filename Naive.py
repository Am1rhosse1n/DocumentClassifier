import numpy as np
# importing necessary libraries for SVM 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import time


X = np.load('X.npy')
Y = np.load('Y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.65,test_size=0.35 , random_state = 0)

start = time.time()

" Naive Bayes MultinomialNB"
nbm = MultinomialNB().fit(X_train, y_train) 
gnb_predictions = nbm.predict(X_test) 
  
# accuracy on X_test 
accuracy_NaiveMultinomial = nbm.score(X_test, y_test) 
  
# creating a confusion matrix 
cm_naiveMultinomial = confusion_matrix(y_test, gnb_predictions) 

end = time.time() - start
print("Multinomial Navie Bayes train time takes " + str(end) + "s")

start = 0
start = time.time()
" Naive Bayes Bernoli"
nbb = BernoulliNB().fit(X_train, y_train) 
gnb_predictions = nbb.predict(X_test) 
  
# accuracy on X_test 
accuracy_NaiveBernoulli = nbb.score(X_test, y_test) 

# creating a confusion matrix 
cm_naiveBernoulli = confusion_matrix(y_test, gnb_predictions) 
end = time.time() - start
print("Bernoli Navie Bayes train time takes " + str(end) + " s")
