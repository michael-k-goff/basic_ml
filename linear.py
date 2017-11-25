# Linear regression

from sklearn import linear_model

def LinearRegression(train_X, train_y, dev_X, dev_y, test_X, test_y):
    # Train the model on the training set
    clf = linear_model.LinearRegression()
    clf = clf.fit(train_X,train_y)
    
    return clf