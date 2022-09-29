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
" KNN Classifier "
# K = 1
knn = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train) 
# accuracy on X_test 
accuracy_KNN_K1 = knn.score(X_test, y_test) 
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm_KNN_K1 = confusion_matrix(y_test, knn_predictions) 

end = time.time() - start
print("KNN classifier with K=1 train time takes " + str(end) + "s")
start = 0
start = time.time()

# K = 3
knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train) 
# accuracy on X_test 
accuracy_KNN_K3 = knn.score(X_test, y_test) 
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm_KNN_K3 = confusion_matrix(y_test, knn_predictions) 

end = time.time() - start
print("KNN classifier with K=3 train time takes " + str(end) + "s")
start = 0
start = time.time()


# K = 5
knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train) 
# accuracy on X_test 
accuracy_KNN_K5 = knn.score(X_test, y_test) 
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
cm_KNN_K5 = confusion_matrix(y_test, knn_predictions)

end = time.time() - start
print("KNN classifier with K=5 train time takes " + str(end) + "s")
 
