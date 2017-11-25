# Review of machine learning algorithms
# Data: from Kaggle, housing prices in King County, Washington from May 2014 to May 2015
# https://www.kaggle.com/harlfoxem/housesalesprediction

# Objective: show how various machine learning algorithms perform in predicting housing prices in terms of other variables.

import os
import pandas as pd
import numpy as np
import sklearn
#import pydot
import load_kc_data
reload(load_kc_data)
import tree
reload(tree)
import svm
reload(svm)
import linear
reload(linear)
import ensemble
reload(ensemble)

os.chdir('My/Path/Here')

train_X, train_y, dev_X, dev_y, test_X, test_y = load_kc_data.load_kc_housing()

# Train models
tree_model = tree.DecisionTree(train_X, train_y, dev_X, dev_y, test_X, test_y)
linear_model = linear.LinearRegression(train_X, train_y, dev_X, dev_y, test_X, test_y)

# Results
def evaluate_model(clf,X,y):
    y_pred = clf.predict(X)
    rms = sklearn.metrics.mean_squared_error(y,y_pred)
    print "The model's RMS is " + str(rms) + ", which is " + str(100*rms/np.var(y)) + "% of data variance."
    
print "Decision Tree:"
evaluate_model(tree_model, test_X, test_y)
print "\nLinear Regression"
evaluate_model(linear_model, test_X, test_y)

# Averaging the results of the two models
ensemble_rms = ensemble.Ensemble(test_X, test_y, [tree_model,linear_model])
print "\nThe ensemble model's RMS is " + str(ensemble_rms) + ", which is " + str(100*ensemble_rms/np.var(test_y)) + "% of data variance."