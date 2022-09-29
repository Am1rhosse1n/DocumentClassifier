import numpy as np
import time
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


X = np.load('X.npy')
Y = np.load('Y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.65,test_size=0.35 , random_state = 0)


" SVM Classifier "
start = time.time()
# training a linear SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
# model accuracy for X_test   
accuracy_linear = svm_model_linear.score(X_test, y_test) 
# creating a confusion matrix 
cm_linear = confusion_matrix(y_test, svm_predictions) 


end = time.time() - start
print("Linear SVM classifier train time takes " + str(end) + "s")

start = 0
start = time.time()


# training a Gaussian SVM classifier
svm_model_Gaussian = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_Gaussian.predict(X_test) 
# model accuracy for X_test   
accuracy_Gaussian = svm_model_Gaussian.score(X_test, y_test) 
  # creating a confusion matrix 
cm_Gaussian = confusion_matrix(y_test, svm_predictions) 
end = time.time() - start
print("Gaussian SVM classifier train time takes " + str(end) + "s")

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


