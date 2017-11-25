# Support Vector Machine
# This model was ultimately not used in the analysis

import numpy as np
import sklearn
from sklearn import svm
from sklearn import preprocessing

def SVMModel(train_X, train_y, dev_X, dev_y, test_X, test_y):
    # Train the model on the training set
    clf = svm.SVR()
    clf = clf.fit(train_X,train_y)

    return clf