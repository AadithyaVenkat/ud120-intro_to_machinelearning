#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import math

features_train, features_test, labels_train, labels_test = preprocess()

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
#print(len(features_train)/100)
#print(len(labels_train)/100)
#features_train = features_train[:round(len(features_train)/100)] 
#labels_train = labels_train[:round(len(labels_train)/100)] 




#########################################################
### your code goes here ###

#########################################################


clf = SVC(C=10000.0, kernel="rbf")
t0 = time()
clf.fit(features_train, labels_train) 
print ("training time:", round(time()-t0, 3), "s")
t1 = time()
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t0, 3), "s")
acc = accuracy_score(pred, labels_test)
print("Accuracy of SVM is",acc)
print("Length of predicted labels are", len(pred))
print(np.unique(pred, return_counts=True))
print(-(2/3)*math.log((2/3),2) - (1/3)*math.log((1/3),2))

